"""Phased-array field model + boundary-element scattering by the printed part.

The total field is linear in the complex transducer drives w (|w_t| <= 1):

    p(x)      = [ a_inc(x) + d(x) @ M ] @ w
    grad p(x) = [ g_inc(x) + dg(x) @ M ] @ w

where M (panels x transducers) is the BEM solution operator: surface pressure
on the scatterer per unit drive.  This is the "modified propagator" idea of
Hirayama et al. 2022 (Sci. Adv.) and lets the hologram solver account for the
part exactly (within the BEM discretisation).
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as sla

from . import constants as C
from .kernels import piston_rows, dl_matrices, winding_number


# ----------------------------------------------------------------------------
# transducer arrays
# ----------------------------------------------------------------------------
def _hex_disc(radius, rings=2):
    """Equal-weight hexagonal sampling of a disc (1+6+12 points for rings=2),
    scaled so the represented area equals pi*radius^2."""
    pts = [(0.0, 0.0)]
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            if (i, j) == (0, 0):
                continue
            if abs(i + j) > rings:
                continue
            pts.append((i + 0.5 * j, np.sqrt(3) / 2 * j))
    pts = np.array(pts)
    h = radius / np.sqrt(len(pts) * (np.sqrt(3) / 2) / np.pi)
    return pts * h


@dataclass
class Panel:
    """A flat PCB carrying an nx x ny grid of transducers."""
    name: str
    center: np.ndarray           # centre of the radiating face [m]
    normal: np.ndarray           # unit vector pointing into the workspace
    u: np.ndarray                # in-plane unit vector (columns)
    nx: int
    ny: int
    pitch: float = C.PITCH
    holes: list = field(default_factory=list)   # [(u_mm, v_mm, radius_mm)] cut-outs

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.normal = np.asarray(self.normal, float) / np.linalg.norm(self.normal)
        u = np.asarray(self.u, float)
        u = u - self.normal * (u @ self.normal)
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.normal, self.u)

    def grid_uv(self):
        us = (np.arange(self.nx) - (self.nx - 1) / 2) * self.pitch
        vs = (np.arange(self.ny) - (self.ny - 1) / 2) * self.pitch
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        uv = np.stack([uu.ravel(), vv.ravel()], 1)
        keep = np.ones(len(uv), bool)
        for (hu, hv, hr) in self.holes:
            keep &= np.hypot(uv[:, 0] - hu, uv[:, 1] - hv) > hr + C.PISTON_RADIUS * 0.6
        return uv[keep]

    def positions(self):
        uv = self.grid_uv()
        return self.center + uv[:, :1] * self.u + uv[:, 1:] * self.v


class Array:
    """Collection of panels -> transducer positions, normals and sub-sources."""

    def __init__(self, panels, vpp=20.0, p_ref_per_vpp=C.P_REF_PER_VPP,
                 piston_radius=C.PISTON_RADIUS, rings=2, name="array"):
        self.name = name
        self.panels = list(panels)
        pos, nrm, pid = [], [], []
        for i, p in enumerate(self.panels):
            xy = p.positions()
            pos.append(xy)
            nrm.append(np.tile(p.normal, (len(xy), 1)))
            pid.append(np.full(len(xy), i))
        self.pos = np.vstack(pos)
        self.nrm = np.vstack(nrm)
        self.panel_id = np.concatenate(pid)
        self.n = len(self.pos)
        self.vpp = vpp
        self.p_ref = p_ref_per_vpp * vpp          # Pa*m on axis at |w|=1
        disc = _hex_disc(piston_radius, rings)
        self.alpha = np.full(len(disc), 1.0 / len(disc))
        sub = np.zeros((self.n, len(disc), 3))
        for i, p in enumerate(self.panels):
            m = self.panel_id == i
            sub[m] = (self.pos[m][:, None, :] + disc[None, :, :1] * p.u + disc[None, :, 1:] * p.v)
        self.sub = np.ascontiguousarray(sub)
        self.k = C.K
        self.gain = np.ones(self.n)       # per-transducer amplitude factor (e.g. loss through a drip screen)

    def set_panel_gain(self, name, g):
        for i, p in enumerate(self.panels):
            if p.name == name:
                self.gain[self.panel_id == i] = g
        return self

    def rows(self, pts, grad=True):
        """Incident propagator rows: A (n,T), G (n,3,T)."""
        pts = np.ascontiguousarray(np.atleast_2d(pts), dtype=float)
        A, G = piston_rows(pts, self.sub, self.alpha, self.k, self.p_ref, grad)
        if not np.all(self.gain == 1.0):
            A = A * self.gain
            if grad:
                G = G * self.gain
        return (A, G) if grad else (A, None)

    def with_reflections(self, R):
        """Copy of this array in which every transducer also radiates via its
        first-order mirror images in the other boards (specular amplitude
        reflection coefficient R).  Models the reverberant field of a closed
        box of boards; higher orders are neglected."""
        import copy
        out = copy.copy(self)
        out.gain = self.gain.copy()
        subs = [self.sub]
        alphas = [self.alpha]
        for j, pan in enumerate(self.panels):
            d = (self.sub - pan.center) @ pan.normal                    # (T,S)
            img = self.sub - 2.0 * d[:, :, None] * pan.normal
            own = (self.panel_id == j)
            img[own] = self.sub[own] + 10.0 * pan.normal * 0 + 1e3     # park own-board images far away
            subs.append(img)
            alphas.append(R * self.alpha)
        out.sub = np.ascontiguousarray(np.concatenate(subs, axis=1))
        out.alpha = np.concatenate(alphas)
        out.name = f"{self.name}_R{R}"
        return out

    def focus_phases(self, x):
        """Phases that make all transducers arrive in phase at x (free space)."""
        d = np.linalg.norm(self.pos - np.asarray(x), axis=1)
        return (-self.k * d) % (2 * np.pi)


# ----------------------------------------------------------------------------
# triangle mesh helper
# ----------------------------------------------------------------------------
class Mesh:
    def __init__(self, verts, faces):
        self.verts = np.asarray(verts, float)
        self.faces = np.asarray(faces, np.int64)
        t = self.verts[self.faces]
        a2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        self.faces = self.faces[a2 > 1e-6 * np.median(a2)]       # drop degenerate panels
        self._derive()
        if self.signed_volume() < 0:           # make normals point outward
            self.faces = self.faces[:, ::-1].copy()
            self._derive()

    def _derive(self):
        self.tri = np.ascontiguousarray(self.verts[self.faces])       # (N,3,3)
        e1 = self.tri[:, 1] - self.tri[:, 0]
        e2 = self.tri[:, 2] - self.tri[:, 0]
        cr = np.cross(e1, e2)
        twice = np.linalg.norm(cr, axis=1)
        self.area = 0.5 * twice
        self.nrm = np.ascontiguousarray(cr / twice[:, None])
        self.cen = np.ascontiguousarray(self.tri.mean(1))
        e3 = self.tri[:, 2] - self.tri[:, 1]
        self.size = np.maximum.reduce([np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1),
                                       np.linalg.norm(e3, axis=1)])
        self.n = len(self.faces)

    def signed_volume(self):
        t = self.tri
        return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0

    def interior_points(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        lo, hi = self.verts.min(0), self.verts.max(0)
        out = []
        for _ in range(60):
            cand = lo + (hi - lo) * rng.random((n * 6, 3))
            wn = winding_number(cand, self.tri)
            out.append(cand[wn > 0.9])
            if sum(len(o) for o in out) >= n:
                break
        pts = np.vstack(out)
        return pts[:n]


# ----------------------------------------------------------------------------
# BEM: sound-hard scatterer
# ----------------------------------------------------------------------------
class Scatterer:
    """Sound-hard body in the field of an Array.

    Direct collocation BEM with constant panels,
        (1/2 I - D) mu = p_inc      on S,
    stabilised against fictitious interior resonances with CHIEF points
    (extra equations  -D_c mu = p_inc(x_c)  at interior points; least squares).
    """

    NEAR = 4.0

    def __init__(self, mesh: Mesh, array: Array, n_chief=40, verbose=False):
        self.mesh = mesh
        self.array = array
        m = mesh
        self.k = array.k
        idx = np.arange(m.n, dtype=np.int64)
        D, _ = dl_matrices(m.cen, m.tri, m.nrm, m.area, m.size, self.k, self.NEAR, idx, False)
        A = 0.5 * np.eye(m.n) - D
        Pinc, _ = array.rows(m.cen, grad=False)
        if n_chief > 0:
            xc = m.interior_points(n_chief)
            none = np.full(len(xc), -1, np.int64)
            Dc, _ = dl_matrices(xc, m.tri, m.nrm, m.area, m.size, self.k, self.NEAR, none, False)
            Pc, _ = array.rows(xc, grad=False)
            A = np.vstack([A, -Dc])
            rhs = np.vstack([Pinc, Pc])
            Q, R = sla.qr(A, mode="economic", overwrite_a=True, check_finite=False)
            self.M = sla.solve_triangular(R, Q.conj().T @ rhs, check_finite=False)
            self.cond = float(np.abs(np.diag(R)).max() / np.abs(np.diag(R)).min())
        else:
            self.M = sla.solve(A, Pinc, check_finite=False)
            self.cond = np.nan
        self.M = np.ascontiguousarray(self.M)
        if verbose:
            print(f"  BEM: {m.n} panels, {array.n} transducers, R-diag ratio {self.cond:.1f}")

    def dl(self, pts, grad=True):
        pts = np.ascontiguousarray(np.atleast_2d(pts), dtype=float)
        none = np.full(len(pts), -1, np.int64)
        m = self.mesh
        return dl_matrices(pts, m.tri, m.nrm, m.area, m.size, self.k, self.NEAR, none, grad)


class Field:
    """Total-field evaluator: free space if scat is None, else incident + scattered."""

    def __init__(self, array: Array, scat: Scatterer = None):
        self.array = array
        self.scat = scat

    def rows(self, pts):
        """Total propagator rows A (n,T), G (n,3,T) such that p = A@w, grad p = G@w."""
        A, G = self.array.rows(pts, grad=True)
        if self.scat is not None:
            Dp, Dg = self.scat.dl(pts, grad=True)
            A = A + Dp @ self.scat.M
            n = Dg.shape[0]
            G = G + (Dg.reshape(n * 3, -1) @ self.scat.M).reshape(n, 3, -1)
        return A, G

    def surface_density(self, w):
        return None if self.scat is None else self.scat.M @ w

    def eval(self, pts, w, mu=None):
        """p (n,), grad p (n,3) for drives w.  Pass mu = surface_density(w) to reuse it."""
        A, G = self.array.rows(pts, grad=True)
        p = A @ w
        g = G @ w
        if self.scat is not None:
            if mu is None:
                mu = self.scat.M @ w
            Dp, Dg = self.scat.dl(pts, grad=True)
            p = p + Dp @ mu
            g = g + Dg @ mu
        return p, g
