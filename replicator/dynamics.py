"""Droplet flight under time-varying holograms, and the delivery planner.

Controller side ("model"): plans a desired droplet trajectory x_d(t) and solves
a hologram per keyframe such that a droplet *on* that trajectory feels exactly
the force the trajectory requires (gravity + inertia + drag feed-forward).

Plant side ("truth"): integrates the droplet in a field that may differ from the
controller's model (drive phase/amplitude errors, phase quantisation, sound
speed, part geometry, air currents).
"""
from dataclasses import dataclass, field as dfield
import numpy as np

from . import constants as C
from .gorkov import Droplet, TrapStencil
from .holo import TrapSolver
from .scene import frame_from_normal


# ----------------------------------------------------------------------------
# hologram sequence
# ----------------------------------------------------------------------------
@dataclass
class Keys:
    t: np.ndarray            # (K,) keyframe times [s]
    phi: np.ndarray          # (K,T) phases
    amp: np.ndarray          # (K,) global drive amplitude
    xd: np.ndarray           # (K,3) desired droplet position
    info: list = dfield(default_factory=list)

    def at(self, t):
        """Linearly interpolated (shortest-arc) phases and amplitude at time t."""
        if t <= self.t[0]:
            return self.phi[0], self.amp[0]
        if t >= self.t[-1]:
            return self.phi[-1], self.amp[-1]
        k = int(np.searchsorted(self.t, t) - 1)
        f = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        d = np.angle(np.exp(1j * (self.phi[k + 1] - self.phi[k])))
        return self.phi[k] + f * d, (1 - f) * self.amp[k] + f * self.amp[k + 1]

    def desired(self, t):
        return np.array([np.interp(t, self.t, self.xd[:, d]) for d in range(3)])


def concat(a: Keys, b: Keys, gap=1e-6):
    """b follows a; b's clock restarts right after a ends."""
    tb = b.t - b.t[0] + a.t[-1] + gap
    return Keys(np.concatenate([a.t, tb]), np.vstack([a.phi, b.phi]),
                np.concatenate([a.amp, b.amp]), np.vstack([a.xd, b.xd]), a.info + b.info)


# ----------------------------------------------------------------------------
# plant
# ----------------------------------------------------------------------------
class Plant:
    """The 'real' machine: true field + drive imperfections + air motion."""

    def __init__(self, field, phase_err=None, amp_err=None, phase_levels=None,
                 wind=None, update_hz=4000.0):
        self.field = field
        n = field.array.n
        self.phase_err = np.zeros(n) if phase_err is None else phase_err
        self.gain = np.ones(n) if amp_err is None else 1.0 + amp_err
        self.phase_levels = phase_levels
        self.wind = np.zeros(3) if wind is None else np.asarray(wind, float)
        self.update_dt = 1.0 / update_hz

    def drives(self, phi, amp):
        if self.phase_levels:
            q = 2 * np.pi / self.phase_levels
            phi = np.round(phi / q) * q
        return amp * self.gain * np.exp(1j * (phi + self.phase_err))


def drag_force(drop: Droplet, vrel):
    """Schiller-Naumann drag on a sphere moving at vrel relative to the air."""
    sp = np.linalg.norm(vrel)
    re = C.RHO0 * sp * drop.d / C.MU_AIR
    return -3 * np.pi * C.MU_AIR * drop.d * (1 + 0.15 * re ** 0.687) * vrel


_FD = None


def acoustic_force(fld, drop, x, w, mu, h=0.03e-3):
    """-grad U by central differences (6 field evaluations).
    Also returns the acoustic Bond number at the droplet (from the stencil mean)."""
    global _FD
    if _FD is None or _FD[0] != h:
        off = np.zeros((6, 3))
        for d in range(3):
            off[2 * d, d] = h; off[2 * d + 1, d] = -h
        _FD = (h, off)
    p, g = fld.eval(x + _FD[1], w, mu)
    U = drop.potential(p, g)
    ba = float(drop.bond_number(p.mean(), g.mean(0)))
    return -(U[0::2] - U[1::2]) / (2 * h), ba


class ShapeMode:
    """Fundamental (l=2, oblate/prolate) shape oscillation of the droplet.

    eps is the P2 amplitude of the surface, r = a (1 + eps P2(cos theta)).
    Small-deformation theory:  eps'' + 2 g eps' + w2^2 (eps - eps_s) = 0 with
        w2^2 = 8 sigma / (rho a^3)        (Rayleigh)
        g    = 5 mu / (rho a^2)           (Lamb; viscous damping)
        eps_s = -(3/32) Ba                (static flattening by the sound field)
    Integrated exactly per step for piecewise-constant forcing.
    """

    def __init__(self, drop: Droplet, dt):
        from scipy.linalg import expm
        self.w2 = np.sqrt(8 * drop.sigma / (drop.rho * drop.a ** 3))
        self.g = 5 * drop.mu / (drop.rho * drop.a ** 2)
        self.E = expm(np.array([[0.0, 1.0], [-self.w2 ** 2, -2 * self.g]]) * dt)
        self.eps = 0.0
        self.deps = 0.0
        self.eps_min = 0.0

    def step(self, ba):
        es = -(3.0 / 32.0) * ba
        y = self.E @ np.array([self.eps - es, self.deps])
        self.eps = es + y[0]; self.deps = y[1]
        self.eps_min = min(self.eps_min, self.eps)

    @property
    def ba_dynamic(self):
        """Static Bond number that would give the worst deformation actually reached."""
        return -(32.0 / 3.0) * self.eps_min

    @property
    def aspect_max(self):
        e = self.eps_min
        return (1 - e / 2) / (1 + e)


def simulate(plant: Plant, drop: Droplet, x0, v0, keys: Keys, part=None, dt=1e-4,
             t_extra=0.03, record_every=5, max_dev=4e-3):
    """Integrate the droplet (velocity Verlet) until it lands, is lost, or time runs out.

    Returns dict(status, t, x, v, traj(t,x,xd), ba_max, ba_dyn, aspect_max).
    status in {'landed', 'lost', 'timeout'}.
    """
    x = np.array(x0, float); v = np.array(v0, float)
    fld = plant.field
    t = keys.t[0]
    t_end = keys.t[-1] + t_extra
    next_update = -np.inf
    w = mu = None
    traj = []
    ba_max = 0.0
    shape = ShapeMode(drop, dt)

    def force(xx, vv):
        fa, ba = acoustic_force(fld, drop, xx, w, mu)
        return fa + drop.mass * C.GVEC + drag_force(drop, vv - plant.wind), ba

    step = 0
    F = None
    status = "timeout"
    while t < t_end:
        if t >= next_update:
            # controller outputs the hologram for the middle of the hold interval
            phi, amp = keys.at(t + 0.5 * plant.update_dt)
            w = plant.drives(phi, amp)
            mu = fld.surface_density(w)
            next_update = t + plant.update_dt
            F, ba = force(x, v)
        a = F / drop.mass
        x_new = x + v * dt + 0.5 * a * dt * dt
        v_half = v + 0.5 * a * dt
        F_new, ba = force(x_new, v_half + 0.5 * a * dt)
        v = v_half + 0.5 * (F_new / drop.mass) * dt
        x = x_new; F = F_new; t += dt; step += 1
        ba_max = max(ba_max, ba)
        shape.step(ba)
        if step % record_every == 0:
            xd = keys.desired(t)
            traj.append((t, *x, *xd))
            if np.linalg.norm(x - xd) > max_dev and t < keys.t[-1]:
                status = "lost"; break
        if part is not None and part.dist(x)[0] <= drop.a:
            status = "landed"; break
        if not np.all(np.isfinite(x)):
            status = "lost"; break
    return dict(status=status, t=t, x=x, v=v, traj=np.array(traj), ba_max=ba_max,
                ba_dyn=shape.ba_dynamic, aspect_max=shape.aspect_max)


# ----------------------------------------------------------------------------
# trajectory primitives
# ----------------------------------------------------------------------------
def min_jerk(L, a_max):
    """Rest-to-rest quintic along a path of length L with peak acceleration a_max.
    Returns duration T and functions s(t), s'(t), s''(t)."""
    T = np.sqrt(5.7735 * L / a_max) if L > 0 else 1e-3

    def s(t):
        u = np.clip(t / T, 0, 1); return L * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)

    def sd(t):
        u = np.clip(t / T, 0, 1); return L * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T

    def sdd(t):
        u = np.clip(t / T, 0, 1); return L * (60 * u - 180 * u ** 2 + 120 * u ** 3) / T ** 2
    return T, s, sd, sdd


class Curve:
    """Smooth path, arc-length parametrised from dense samples.

    pos(s), tangent(s) = dP/ds (unit), curv(s) = d2P/ds2 (curvature vector), so
        v = tangent * s',   a = curv * s'^2 + tangent * s''.
    """

    def __init__(self, pts):
        p = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        self.L = cum[-1]
        self.s = np.linspace(0, self.L, max(400, len(p)))
        self.p = np.stack([np.interp(self.s, cum, p[:, d]) for d in range(3)], 1)
        ds = self.s[1] - self.s[0]
        self.d1 = np.gradient(self.p, ds, axis=0)
        self.d1 /= np.linalg.norm(self.d1, axis=1)[:, None]
        self.d2 = np.gradient(self.d1, ds, axis=0)

    def _ip(self, arr, s):
        return np.array([np.interp(s, self.s, arr[:, d]) for d in range(3)])

    def pos(self, s): return self._ip(self.p, s)
    def tangent(self, s): return self._ip(self.d1, s)
    def curv(self, s): return self._ip(self.d2, s)

    @staticmethod
    def line(a, b):
        return Curve(np.linspace(np.asarray(a, float), np.asarray(b, float), 50))

    @staticmethod
    def bezier(p0, p3, dir_out=None, dir_in=None, pull=0.4):
        """Cubic Bezier leaving p0 along dir_out and arriving at p3 travelling along dir_in."""
        p0 = np.asarray(p0, float); p3 = np.asarray(p3, float)
        L = np.linalg.norm(p3 - p0)
        d = (p3 - p0) / L
        do = d if dir_out is None else np.asarray(dir_out, float) / np.linalg.norm(dir_out)
        di = d if dir_in is None else np.asarray(dir_in, float) / np.linalg.norm(dir_in)
        p1 = p0 + do * pull * L
        p2 = p3 - di * pull * L
        u = np.linspace(0, 1, 1500)[:, None]
        return Curve((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3)


# ----------------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------------
class Planner:
    def __init__(self, field, drop: Droplet, ba_max=C.BA_DESIGN, ds=0.3e-3,
                 a_max=0.5 * C.G_ACC, w_smooth=3.0, verbose=False):
        self.field = field
        self.drop = drop
        self.solver = TrapSolver(field, drop, ba_max=ba_max, w_smooth=w_smooth)
        self.ds = ds
        self.a_max = a_max
        self.tau_v = 2 * drop.rho * drop.a ** 2 / (9 * C.MU_AIR)     # Stokes relaxation time
        self.wind = np.zeros(3)        # measured steady air motion (digital-twin trap test); 0 = still air
        self.verbose = verbose

    # -- feed-forward: net force (in weights) a droplet on the trajectory needs
    def _ff(self, vel, acc):
        return (acc + (vel - self.wind) / self.tau_v) / C.G_ACC

    def _solve(self, x, f_target, phi_prev, s_prev, fresh=False, maxiter=250):
        sv = self.solver
        sv.f_target = np.asarray(f_target, float)
        if fresh or phi_prev is None:
            return sv.solve_best(x, n_random=1, extra=None if phi_prev is None else {"prev": phi_prev})
        return sv.solve(x, phi_prev, s_prev, maxiter=maxiter, phi_ref=phi_prev)

    def transit(self, path: Curve, phi0=None, s0=0.5, t0=0.0):
        """Carry the droplet along `path`, rest to rest, in a 3-D trap."""
        self.solver.set_mode()
        T, s, sd, sdd = min_jerk(path.L, self.a_max)
        n = max(2, int(np.ceil(path.L / self.ds)) + 1)
        tt = np.linspace(0, T, 4000)
        ts = np.interp(np.linspace(0, path.L, n), s(tt), tt)
        phis, amps, xs, infos = [], [], [], []
        phi, amp = phi0, s0
        for i, t in enumerate(ts):
            x = path.pos(s(t)); tan = path.tangent(s(t))
            ff = self._ff(tan * sd(t), tan * sdd(t) + path.curv(s(t)) * sd(t) ** 2)
            phi, amp, info = self._solve(x, ff, phi, amp, fresh=(i == 0 and phi0 is None))
            phis.append(phi); amps.append(amp); xs.append(x); infos.append(info)
            if self.verbose and i % 10 == 0:
                print(f"    transit {i:3d}/{n} kappa={np.round(info['kappa'],2)} Ba={info['Ba']:.2f} "
                      f"amp={amp:.2f} |f_res|={np.linalg.norm(info['f_res']):.3f}")
        return Keys(t0 + ts, np.array(phis), np.array(amps), np.array(xs), infos)

    def follow(self, ts, X, V, A, phi0=None, s0=0.5, t0=0.0):
        """Holograms for an arbitrary time-parametrised trajectory (positions X,
        velocities V, accelerations A at times ts).  Each hologram supplies exactly
        the net force a droplet on the trajectory needs: m a minus drag."""
        self.solver.set_mode()
        phis, amps, infos = [], [], []
        phi, amp = phi0, s0
        for i in range(len(ts)):
            ff = (A[i] - drag_force(self.drop, V[i] - self.wind) / self.drop.mass) / C.G_ACC
            phi, amp, info = self._solve(X[i], ff, phi, amp, fresh=(i == 0 and phi0 is None))
            phis.append(phi); amps.append(amp); infos.append(info)
        return Keys(t0 + np.asarray(ts), np.array(phis), np.array(amps), np.asarray(X), infos)

    def hold(self, x, duration, phi0, s0, n=2):
        self.solver.set_mode()
        phi, amp, info = self._solve(x, self._ff(np.zeros(3), np.zeros(3)), phi0, s0)
        t = np.linspace(0, duration, n)
        return Keys(t, np.tile(phi, (n, 1)), np.full(n, amp), np.tile(x, (n, 1)), [info] * n)

    def press(self, xs, normal, gap0, push=1.0, phi0=None,
              gaps=(0.03e-3, 0.3e-3, 0.6e-3, 0.9e-3, 1.2e-3), n_surf=None):
        """Final approach with one static funnel hologram (see holo.solve_funnel).

        The hologram is simply held; the droplet accelerates onto the surface at
        ~`push` weights and lands whenever it gets there.
        """
        fr = frame_from_normal(normal)
        a = self.drop.a
        extra = None if phi0 is None else {"prev": phi0}
        phi, amp, info = self.solver.solve_funnel(xs, fr, np.asarray(gaps), push=push, extra=extra, n_surf=n_surf)
        t_nom = np.sqrt(2 * gap0 / (push * C.G_ACC))
        t = np.array([0.0, t_nom, 3.0 * t_nom])
        c0 = xs + a * (fr[2] if n_surf is None else np.asarray(n_surf, float))
        xd = np.array([c0 + fr[2] * gap0, c0, c0])
        if self.verbose:
            print(f"    funnel [{info['init']}]: f_n={np.round(info['f_n'],2)} kappa_t={np.round(info['kappa_t'],2)} "
                  f"Ba={info['Ba']:.2f} amp={amp:.2f}")
        return Keys(t, np.tile(phi, (3, 1)), np.full(3, amp), xd, [info] * 3)
