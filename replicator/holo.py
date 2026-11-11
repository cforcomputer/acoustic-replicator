"""Hologram solver: find transducer phases (and one global drive amplitude) that
put a stable trap, balanced against gravity, at a requested point -- with the
scattering of the part folded into the propagators, and subject to a limit on
the acoustic Bond number so the droplet is not flattened / atomised.

Two kinds of hologram are produced:

trap    a 3-D stable trap at one point, with a prescribed net force there
        (gravity balance + feed-forward for the planned motion);
funnel  a single static hologram for the final approach: at several stand-off
        distances along the site normal it pushes the droplet toward the
        surface with a set force while centring it tangentially.  Because it
        does not depend on where the droplet is along the normal, it is immune
        to timing errors.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from . import constants as C
from .gorkov import TrapStencil, Droplet

MM = 1e-3


@dataclass
class Term:
    st: TrapStencil
    Pm: np.ndarray           # (m,3) directions in which stiffness is rewarded
    f_target: np.ndarray     # desired net force / weight at the stencil centre
    weight: float = 1.0
    axis: np.ndarray = None  # if set: force along -axis only needs to lie in [lo, hi] weights
    lo: float = 0.5
    hi: float = 2.5


class TrapSolver:
    """Normalised quantities
        kappa_i = eig_i(Pm H Pm^T) * 1 mm / weight   (restoring force, weights per mm)
        f       = (F_acoustic + m g) / weight        (net force at the point)
        Ba      = acoustic Bond number at the droplet
    Objective (minimised) over phases phi_t and global amplitude s in [s_min, 1]:
        J = sum_terms w [ -softmin_tau(kappa) + w_force |f - f_target|^2
                          + w_bond max(0, Ba/Ba_max - 1)^2 ]
            + w_smooth * mean(1 - cos(phi - phi_ref))
    """

    def __init__(self, field, drop: Droplet, h=0.05e-3, w_force=200.0, tau=0.3,
                 ba_max=C.BA_DESIGN, w_bond=400.0, gvec=C.GVEC, s_min=0.02, w_smooth=0.0):
        self.field = field
        self.drop = drop
        self.h = h
        self.w_force = w_force
        self.tau = tau
        self.ba_max = ba_max
        self.w_bond = w_bond
        self.ghat = np.asarray(gvec) / C.G_ACC
        self.Pm = np.eye(3)
        self.f_target = np.zeros(3)
        self.s_min = s_min
        self.w_smooth = w_smooth

    def set_mode(self, frame=None, scale=(1.0, 1.0, 1.0)):
        """Stiffness is rewarded along the rows of `frame`, weighted by `scale`."""
        R = np.eye(3) if frame is None else np.asarray(frame, float)
        self.Pm = np.diag(scale) @ R
        return self

    # ------------------------------------------------------------------
    def _term(self, w, s, term: Term):
        st = term.st
        d = self.drop
        W = d.weight
        U, P, Gv = st.potentials(w)
        F = st.Fop @ U
        h6 = st.Hop @ U
        H = np.array([[h6[0], h6[3], h6[4]], [h6[3], h6[1], h6[5]], [h6[4], h6[5], h6[2]]])
        lam, V = np.linalg.eigh(term.Pm @ H @ term.Pm.T)
        kap = lam * MM / W
        z = -kap / self.tau
        zmax = z.max()
        ez = np.exp(z - zmax)
        sm = -self.tau * (zmax + np.log(ez.sum()))
        soft = ez / ez.sum()
        coeff = np.zeros(19)
        if term.axis is None:
            f = F / W + self.ghat - term.f_target
            J = -sm + self.w_force * (f @ f)
            coeff += 2 * self.w_force * (f @ st.Fop) / W
        else:
            # funnel: zero force across the axis; along it, any inward push in [lo, hi]
            ax = term.axis
            f = F / W + self.ghat
            fn = f @ ax
            ft = f - fn * ax
            under = max(0.0, fn + term.lo)          # not pushing in hard enough
            over = max(0.0, -fn - term.hi)          # pushing in too hard
            J = -sm + self.w_force * (ft @ ft + under ** 2 + over ** 2)
            coeff += 2 * self.w_force * ((ft @ st.Fop) + (under - over) * (ax @ st.Fop)) / W
        for i in range(len(lam)):
            v = term.Pm.T @ V[:, i]
            c6 = np.array([v[0] ** 2, v[1] ** 2, v[2] ** 2,
                           2 * v[0] * v[1], 2 * v[0] * v[2], 2 * v[1] * v[2]])
            coeff -= soft[i] * (c6 @ st.Hop) * MM / W
        gphi = st.dU_dphi(w, P, Gv, coeff)
        gs = 2.0 * (coeff @ U) / s
        if self.ba_max is not None:
            ep = np.abs(P[0]) ** 2 / (2 * C.RHO0 * C.C0 ** 2)
            ek = np.sum(np.abs(Gv[0]) ** 2) / (2 * C.RHO0 * C.OMEGA ** 2)
            ba = 2.0 * max(ep, ek) * d.a / d.sigma
            ex = ba / self.ba_max - 1.0
            if ex > 0:
                J += self.w_bond * ex ** 2
                pref = 2 * self.w_bond * ex / self.ba_max * 2.0 * d.a / d.sigma
                if ep >= ek:
                    gq = np.conj(P[0]) * st.A[0] / (2 * C.RHO0 * C.C0 ** 2)
                else:
                    gq = np.einsum("d,dt->t", np.conj(Gv[0]), st.G[0]) / (2 * C.RHO0 * C.OMEGA ** 2)
                gphi = gphi + pref * (-2.0 * np.imag(w * gq))
                gs += 2 * self.w_bond * ex * 2 * ba / (self.ba_max * s)
        return J, gphi, gs

    def _objective(self, xvar, terms, phi_ref):
        phi = xvar[:-1]
        s = xvar[-1]
        w = s * np.exp(1j * phi)
        J = 0.0
        grad = np.zeros_like(xvar)
        for term in terms:
            j, gphi, gs = self._term(w, s, term)
            J += term.weight * j
            grad[:-1] += term.weight * gphi
            grad[-1] += term.weight * gs
        if phi_ref is not None and self.w_smooth > 0:
            dphi = phi - phi_ref
            J += self.w_smooth * np.mean(1 - np.cos(dphi))
            grad[:-1] += self.w_smooth * np.sin(dphi) / len(phi)
        return J, grad

    def _run(self, terms, phi0, s0, maxiter, phi_ref=None):
        x0 = np.append(phi0, s0)
        bounds = [(None, None)] * len(phi0) + [(self.s_min, 1.0)]
        res = minimize(self._objective, x0, args=(terms, phi_ref), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=maxiter, maxcor=30, ftol=1e-11, gtol=1e-8))
        return np.mod(res.x[:-1], 2 * np.pi), float(res.x[-1]), float(res.fun), int(res.nit)

    # ------------------------------------------------------------------ single trap
    def solve(self, x, phi0, s0=0.5, maxiter=400, stencil=None, phi_ref=None):
        st = stencil if stencil is not None else TrapStencil(self.field, x, self.drop, self.h)
        term = Term(st, self.Pm, self.f_target)
        phi, s, J, nit = self._run([term], phi0, s0, maxiter, phi_ref)
        info = self.diagnose(st, phi, s)
        info["J"] = J
        info["nit"] = nit
        return phi, s, info

    def diagnose(self, st, phi, s=1.0, Pm=None, f_target=None):
        Pm = self.Pm if Pm is None else Pm
        f_target = self.f_target if f_target is None else f_target
        w = s * np.exp(1j * phi)
        m = st.metrics(w)
        W = self.drop.weight
        lam, V = np.linalg.eigh(m["H"])
        f = m["F"] / W + self.ghat
        lam_s = np.linalg.eigvalsh(Pm @ m["H"] @ Pm.T)
        try:
            off = -np.linalg.solve(m["H"], (f - f_target) * W)   # where the force target is met
        except np.linalg.LinAlgError:
            off = np.full(3, np.nan)
        return dict(kappa=lam * MM / W, kappa_sub=lam_s * MM / W, f_net=f, axes=V,
                    f_res=f - f_target, offset=off, p0=abs(m["p"]),
                    Ba=m["Ba"], amp=s, stable=bool(lam.min() > 0), H=m["H"],
                    freq_hz=np.sqrt(np.maximum(lam, 0) / self.drop.mass) / (2 * np.pi))

    def initial_guesses(self, x, n_random=2, seed=0):
        """Classical trap signatures applied on top of focusing phases."""
        arr = self.field.array
        foc = arr.focus_phases(x)
        rel = arr.pos - np.asarray(x)
        out = {"focus": foc}
        for nm, ax in (("twin_x", 0), ("twin_y", 1), ("twin_z", 2)):
            out[nm] = foc + np.pi * (rel[:, ax] > 0)
        out["vortex_z"] = foc + np.arctan2(rel[:, 1], rel[:, 0])
        out["vortex_x"] = foc + np.arctan2(rel[:, 2], rel[:, 1])
        rng = np.random.default_rng(seed)
        for i in range(n_random):
            out[f"rand{i}"] = rng.uniform(0, 2 * np.pi, arr.n)
        return out

    def solve_best(self, x, maxiter=400, extra=None, n_random=2, seed=0, s0=0.5):
        st = TrapStencil(self.field, x, self.drop, self.h)
        best = None
        guesses = self.initial_guesses(x, n_random, seed)
        if extra:
            guesses.update(extra)
        for nm, g in guesses.items():
            phi, s, info = self.solve(x, g, s0, maxiter, stencil=st)
            info["init"] = nm
            if best is None or info["J"] < best[2]["J"]:
                best = (phi, s, info)
        return best

    # ------------------------------------------------------------------ funnel
    def solve_funnel(self, xs, frame, gaps, push=1.0, extra=None, n_random=3, maxiter=500, seed=0,
                     n_surf=None, lo=0.5, hi=2.5):
        """One static hologram that, at every stand-off in `gaps` along the
        approach axis frame[2], gives net force = -push*axis (zero across it)
        and stiffness across the axis.  The axis passes through the droplet's
        touchdown position xs + a*n_surf; n_surf defaults to the axis itself
        (approach along the surface normal).  Returns phi, s, info."""
        fr = np.asarray(frame, float)
        a = self.drop.a
        c0 = xs + a * (fr[2] if n_surf is None else np.asarray(n_surf, float))
        Pm = fr[:2]
        terms = []
        for g in gaps:
            st = TrapStencil(self.field, c0 + fr[2] * g, self.drop, self.h)
            terms.append(Term(st, Pm, -push * fr[2], 1.0 / len(gaps), axis=fr[2], lo=lo, hi=hi))
        guesses = self.initial_guesses(c0 + fr[2] * gaps[0], n_random, seed)
        if extra:
            guesses.update(extra)
        best = None
        for nm, g0 in guesses.items():
            phi, s, J, nit = self._run(terms, g0, 0.5, maxiter)
            # rank by the weakest centring stiffness anywhere along the funnel, subject to an inward push
            dg = [self.diagnose(t.st, phi, s, Pm, t.f_target) for t in terms]
            kmin = min(d["kappa_sub"].min() for d in dg)
            fmax = max(d["f_net"] @ fr[2] for d in dg)
            score = kmin - 20.0 * max(0.0, fmax + 0.3)
            if best is None or score > best[4]:
                best = (phi, s, J, nm, score)
        phi, s, J, nm, _ = best
        per = [self.diagnose(t.st, phi, s, Pm, t.f_target) for t in terms]
        info = dict(J=J, init=nm, gaps=np.array(gaps),
                    f_n=np.array([p["f_net"] @ fr[2] for p in per]),
                    f_t=np.array([np.linalg.norm(p["f_net"] - (p["f_net"] @ fr[2]) * fr[2]) for p in per]),
                    kappa_t=np.array([p["kappa_sub"].min() for p in per]),
                    Ba=max(p["Ba"] for p in per), amp=s, per=per)
        return phi, s, info


def restoring_profile(field, drop, x, w, directions, reach=4e-3, n=81, gvec=C.GVEC):
    """Net force (acoustic + gravity) along each direction, sampled on a line
    through x, in units of droplet weight.  Returns s (n,), f (ndir, n)."""
    s = np.linspace(-reach, reach, n)
    h = 0.05e-3
    out = []
    mu = field.surface_density(w)
    for dvec in directions:
        dvec = np.asarray(dvec, float) / np.linalg.norm(dvec)
        pts = x + s[:, None] * dvec
        pp, gp = field.eval(pts + h * dvec, w, mu)
        pm, gm = field.eval(pts - h * dvec, w, mu)
        F = -(drop.potential(pp, gp) - drop.potential(pm, gm)) / (2 * h)
        out.append((F + drop.mass * (np.asarray(gvec) @ dvec)) / drop.weight)
    return s, np.array(out)


def force_at(field, drop, pts, w, gvec=C.GVEC, h=0.03e-3):
    """Net force (acoustic + gravity) in weights at each point (n,3)."""
    pts = np.atleast_2d(pts)
    mu = field.surface_density(w)
    F = np.zeros_like(pts)
    for d in range(3):
        e = np.zeros(3); e[d] = h
        pp, gp = field.eval(pts + e, w, mu)
        pm, gm = field.eval(pts - e, w, mu)
        F[:, d] = -(drop.potential(pp, gp) - drop.potential(pm, gm)) / (2 * h)
    return F / drop.weight + np.asarray(gvec) / C.G_ACC


def holding_force(s, f):
    """Largest restoring force (in weights) the trap can supply on each side
    before the droplet escapes: (toward -s side, toward +s side)."""
    pos = -f[s > 0]
    neg = f[s < 0][::-1]

    def first_peak(v):
        bad = np.where(v < 0)[0]
        end = bad[0] if len(bad) else len(v)
        return float(v[:end].max()) if end > 0 else 0.0
    return first_peak(neg), first_peak(pos)
