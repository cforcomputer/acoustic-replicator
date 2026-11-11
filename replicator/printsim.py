"""Droplet-by-droplet print process: planning order, landing, spreading, curing.

Multi-fidelity: every droplet is deposited with the geometric/capillary model
here (cheap); landing errors are drawn from the statistics measured by the full
acoustic Monte-Carlo, and the full physics is re-run on the *actual* growing
part at checkpoints (studies/s6_print.py).
"""
import heapq
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates, binary_fill_holes
from scipy.special import ndtri

from . import constants as C
from .geometry import VoxelPart
from .gorkov import Droplet
from .scene import POLE_RADIUS, frame_from_normal


# ----------------------------------------------------------------------------
# spreading of the landed droplet before the UV flash (viscosity matters here)
# ----------------------------------------------------------------------------
def _g(theta):
    """Contact radius / droplet radius of a spherical cap of fixed volume."""
    return (4 * np.sin(theta) ** 3 / (2 - 3 * np.cos(theta) + np.cos(theta) ** 3)) ** (1 / 3)


def cure_contact_angle(drop: Droplet, t_delay, theta_e=np.radians(25), ln_lambda=10.0):
    """Apparent contact angle when the UV flash arrives, t_delay after touchdown.

    Cox-Voinov / Tanner spreading of a fixed-volume cap toward the equilibrium
    angle theta_e:   U = sigma (theta^3 - theta_e^3) / (9 mu ln_lambda),
    with the contact radius r = a g(theta).  The sub-millisecond inertial stage
    is neglected (start from 150 deg).
    """
    th = np.radians(150.0)
    t = 0.0
    k = drop.sigma / (9 * drop.mu * ln_lambda)
    while t < t_delay and th > theta_e * 1.02:
        U = k * (th ** 3 - theta_e ** 3)
        dth = 1e-3
        dr = drop.a * (_g(th - dth) - _g(th))          # > 0: radius grows as angle falls
        dt = dr / U
        if t + dt > t_delay:
            th -= dth * (t_delay - t) / dt
            break
        th -= dth; t += dt
    return th


def cap_sphere_radius(a, theta):
    """Radius of the sphere whose cap (contact angle theta on a plane) has the droplet's volume."""
    return a * (4 / (2 - 3 * np.cos(theta) + np.cos(theta) ** 3)) ** (1 / 3)


def cap_mean_thickness(a, theta, r_frac=0.75):
    """Mean thickness of the cured cap over a disc of r_frac x its footprint radius."""
    R = cap_sphere_radius(a, theta)
    rho = np.linspace(0, r_frac * R * np.sin(theta), 200)
    t = np.sqrt(R * R - rho * rho) - R * np.cos(theta)
    return float((t * rho).sum() / rho.sum())


def true_sdf(sdf, res):
    """Signed distance to the OUTER surface of a printed part.

    The print's own field is a running minimum over droplets, so deep inside the
    solid it only knows the depth within one droplet, and enclosed bubbles count
    as surface.  Fill the bubbles, recompute the distance from the occupancy, and
    keep the original sub-voxel values next to the surface."""
    occ = binary_fill_holes(sdf < 0)
    d = np.where(occ, -(distance_transform_edt(occ) - 0.5), distance_transform_edt(~occ) - 0.5).astype(np.float32) * res
    near = (np.abs(sdf) < res) & (np.abs(d) < 1.5 * res)
    d[near] = sdf[near]
    return d


# ----------------------------------------------------------------------------
# site lattice + print order
# ----------------------------------------------------------------------------
class Lattice:
    """Hexagonal (ABAB) lattice of deposition sites inside the target shape.

    Layers are horizontal and counted downward from the pole tip.  The cell
    volume equals the cured droplet volume, so a filled lattice reproduces the
    target volume.
    """

    def __init__(self, target, drop: Droplet, layer_h, z_top, shrink=0.05, inset=0.25, protect=False, shell=None, mask=None):
        """inset: keep only sites deeper than inset x pitch below the target surface (the
        remaining shell is left for a conformal skin).  protect=True also keeps shallow
        sites that have no deep site next to them, so walls thinner than the shell still
        get a core to build the skin on.  shell: keep only sites within this depth of the
        surface (the outer shell of the part)."""
        self.vd = drop.vol * (1 - shrink)
        self.h = layer_h
        self.s = np.sqrt(self.vd / (np.sqrt(3) / 2 * layer_h))
        lo, hi = target.bounds()
        pts, lay = [], []
        nl = int(np.ceil((z_top - lo[2]) / layer_h)) + 1
        s = self.s
        for k in range(nl):
            z = z_top - (k + 0.5) * layer_h
            off = np.array([0.0, 0.0]) if k % 2 == 0 else np.array([s / 2, s / (2 * np.sqrt(3))])
            ni = int(np.ceil((hi[0] - lo[0]) / s)) + 3
            nj = int(np.ceil((hi[1] - lo[1]) / (s * np.sqrt(3) / 2))) + 3
            i, j = np.meshgrid(np.arange(-1, ni), np.arange(-1, nj), indexing="ij")
            x = lo[0] + (i + 0.5 * (j % 2)) * s + off[0]
            y = lo[1] + j * s * np.sqrt(3) / 2 + off[1]
            p = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1)
            dp = target.dist(p)
            keep = dp <= (0.0 if protect else -inset * s)
            if shell is not None:
                keep &= dp > -shell
            if mask is not None:                             # extra region test, e.g. "near a sharp edge" / "not near one"
                keep &= mask(p)
            pts.append(p[keep]); lay.append(np.full(keep.sum(), k))
        self.pts = np.vstack(pts)
        self.layer = np.concatenate(lay)
        self.n_core = 0
        if protect and inset > 0:
            deep = target.dist(self.pts) <= -inset * s
            r_nb = 1.02 * max(s, np.sqrt(layer_h ** 2 + s ** 2 / 3))
            near_deep = np.zeros(len(self.pts), bool)
            if deep.any():
                dd, _ = cKDTree(self.pts[deep]).query(self.pts, k=1)
                near_deep = dd <= r_nb
            keep = deep | ~near_deep
            self.n_core = int((keep & ~deep).sum())
            self.pts, self.layer = self.pts[keep], self.layer[keep]
        self.n = len(self.pts)
        self.tree = cKDTree(self.pts)
        r_nb = 1.02 * max(s, np.sqrt(layer_h ** 2 + s ** 2 / 3))
        self.nbrs = self.tree.query_ball_point(self.pts, r_nb)

    def order(self, seeds):
        """Connectivity-respecting order: always the highest available layer first,
        and a site only becomes available once a neighbour has been deposited."""
        done = np.zeros(self.n, bool)
        queued = np.zeros(self.n, bool)
        heap = []
        cnt = 0
        for sidx in seeds:
            heapq.heappush(heap, (int(self.layer[sidx]), cnt, int(sidx))); queued[sidx] = True; cnt += 1
        out = []
        while heap:
            _, _, i = heapq.heappop(heap)
            done[i] = True
            out.append(i)
            for j in self.nbrs[i]:
                if not queued[j]:
                    queued[j] = True
                    heapq.heappush(heap, (int(self.layer[j]), cnt, int(j))); cnt += 1
        return np.array(out), done


class MixedLattice:
    """Coarse droplets for the core of the part, fine droplets for the outer shell.

    Both lattices conserve volume (one cell = one cured droplet), so the pair still
    reproduces the target volume; the fine shell follows the surface in smaller steps.
    `layer` is the depth below the pole tip in units of `dz`, so one ordering runs
    through both."""

    def __init__(self, core, fine, z_top, dz=0.05e-3):
        self.core, self.fine = core, fine
        self.pts = np.vstack([core.pts, fine.pts])
        self.cls = np.concatenate([np.zeros(core.n, np.int8), np.ones(fine.n, np.int8)])
        self.n = len(self.pts)
        self.s = core.s
        self.vd = core.vd
        self.h = np.where(self.cls == 0, core.h, fine.h)
        self.layer = np.floor((z_top - self.pts[:, 2]) / dz).astype(int)
        self.tree = cKDTree(self.pts)
        rad = np.where(self.cls == 0, 1.02 * max(core.s, np.sqrt(core.h ** 2 + core.s ** 2 / 3)),
                       1.02 * max(fine.s, np.sqrt(fine.h ** 2 + fine.s ** 2 / 3)))
        nb = [set(x) for x in self.tree.query_ball_point(self.pts, rad)]
        for i, x in enumerate(nb):                     # make the neighbour relation symmetric
            x.discard(i)
            for j in x:
                nb[j].add(i)
        self.nbrs = [sorted(x) for x in nb]

    def order(self, seeds):
        done = np.zeros(self.n, bool)
        stack = list(seeds)
        done[stack] = True
        while stack:
            i = stack.pop()
            for j in self.nbrs[i]:
                if not done[j]:
                    done[j] = True; stack.append(j)
        return None, done


# ----------------------------------------------------------------------------
# the growing part
# ----------------------------------------------------------------------------
class PrintedPart(VoxelPart):
    """Voxel SDF of pole + cured droplets, with volume-conserving blob deposition."""

    def __init__(self, lo, hi, res=0.05e-3, band=0.5e-3):
        super().__init__(lo, hi, res=res, band=band)

    def _local(self, c, r):
        i0, i1 = self._box(c - r - self.res, c + r + self.res)
        X, Y, Z = self._local_grid(i0, i1)
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r
        sl = tuple(slice(a, b) for a, b in zip(i0, i1))
        return sl, d

    def added_volume(self, c, r):
        sl, d = self._local(c, r)
        old = self.sdf[sl]
        occ_old = np.clip(0.5 - old / self.res, 0, 1)
        occ_new = np.clip(0.5 - np.minimum(old, d) / self.res, 0, 1)
        return float((occ_new - occ_old).sum()) * self.res ** 3

    def deposit(self, contact, normal, theta_c, volume, a):
        """Add a cured blob where a droplet of radius `a` landed.

        The liquid forms a spherical cap meeting the local surface at contact
        angle theta_c: a sphere of radius r centred r*cos(theta_c) *inside* the
        surface along the normal.  r is solved so that exactly `volume` is added
        outside the existing solid.  Consequences, all physical:
          flat surface  -> the textbook cap;
          groove/valley -> more of the sphere is already solid, so r grows and
                           the liquid spreads to fill the low spot;
          rim/edge      -> less solid under it, so r shrinks to a compact bead
                           (it cannot spread past the edges).
        Returns (centre, r)."""
        ct = np.cos(theta_c)
        lo_r, hi_r = 0.5 * a, 5.0 * a
        for _ in range(16):
            mid = 0.5 * (lo_r + hi_r)
            if self.added_volume(contact - normal * mid * ct, mid) < volume:
                lo_r = mid
            else:
                hi_r = mid
        r = 0.5 * (lo_r + hi_r)
        c = contact - normal * r * ct
        i0, i1 = self._box(c - r - self.band, c + r + self.band)
        X, Y, Z = self._local_grid(i0, i1)
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r
        self._merge(i0, i1, d)
        self.n_deposits += 1
        return c, r

    def deposit_capillary(self, contact, normal, theta_c, volume, a, fine=None, half=0.6e-3, iters=8, blur=1.5):
        """Add a cured droplet whose shape is the capillary equilibrium on the
        *actual* local geometry, meeting the solid at contact angle theta_c.

        1. Start from the analytic cap (sphere of radius r centred r cos(theta)
           inside the surface, r solved for the droplet volume).
        2. Relax with volume-conserving threshold dynamics for wetting (MBO
           scheme, Xu, Wang & Wang 2017) on a fine local grid: blur
               chi_liquid + (1 + cos theta)/2 * chi_solid
           and keep the best-ranked non-solid voxels up to the droplet volume.
           Liquid may only advance where it already touches liquid, so it cannot
           jump to a distant wall.  This descends the surface energy at constant
           volume: liquid wicks into notches between neighbouring beads, bridges
           small gaps and rounds off on edges.
        Returns (centroid, equivalent radius)."""
        h = fine or self.res / 2
        n = int(np.ceil(half / h))
        ct = np.cos(theta_c)
        lo_r, hi_r = 0.5 * a, 5.0 * a
        for _ in range(12):
            mid = 0.5 * (lo_r + hi_r)
            if self.added_volume(contact - normal * mid * ct, mid) < volume:
                lo_r = mid
            else:
                hi_r = mid
        r = 0.5 * (lo_r + hi_r)
        cs = contact - normal * r * ct
        c0 = contact + normal * 0.3 * a
        ax = [c0[d] + h * np.arange(-n, n + 1) for d in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        idx = np.stack([(X - self.lo[0]) / self.res, (Y - self.lo[1]) / self.res, (Z - self.lo[2]) / self.res])
        sd = map_coordinates(self.sdf, idx, order=1, mode="nearest")
        free = sd >= 0
        nfree = np.flatnonzero(free.ravel())
        k = min(int(round(volume / h ** 3)), len(nfree) - 1)
        rank = (r - np.sqrt((X - cs[0]) ** 2 + (Y - cs[1]) ** 2 + (Z - cs[2]) ** 2)).ravel()[nfree]
        liq = np.zeros(X.size, bool)
        liq[nfree[np.argpartition(rank, len(rank) - k)[len(rank) - k:]]] = True
        liq = liq.reshape(X.shape)
        sol = (~free).astype(np.float32) * (0.5 * (1.0 + ct))
        for _ in range(iters):
            gl = gaussian_filter(liq.astype(np.float32), blur, mode="nearest")
            psi = (gl + gaussian_filter(sol, blur, mode="nearest")).ravel()[nfree]
            psi[gl.ravel()[nfree] < 0.04] = -1.0             # only voxels next to existing liquid may fill
            new = np.zeros(X.size, bool)
            new[nfree[np.argpartition(psi, len(psi) - k)[len(psi) - k:]]] = True
            new = new.reshape(X.shape)
            changed = (new ^ liq).sum()
            liq = new
            if changed < 0.01 * k:
                break
        # Signed distance of the liquid with sub-voxel accuracy.  The liquid is first
        # extended into the solid it touches (this does not change the union, but it
        # stops the wall from clipping the thin edge of the cap), then the distance
        # near the interface is read off a unit-width Gaussian blur of that region.
        gl = gaussian_filter(liq.astype(np.float32), blur, mode="nearest")
        ext = liq | ((~free) & (gl > 0.08))
        d = (distance_transform_edt(~ext) - distance_transform_edt(ext)) * h
        d = np.where(ext, d + 0.5 * h, d - 0.5 * h)
        g1 = gaussian_filter(ext.astype(np.float32), 1.0, mode="nearest")
        near = np.abs(d) < 2.2 * h
        d[near] = -ndtri(np.clip(g1[near], 1e-5, 1 - 1e-5)) * h
        i0, i1 = self._box(c0 - (n - 2) * h, c0 + (n - 2) * h)
        gx = [self.lo[dd] + self.res * np.arange(i0[dd], i1[dd]) for dd in range(3)]
        GX, GY, GZ = np.meshgrid(*gx, indexing="ij")
        loc = np.stack([(GX - ax[0][0]) / h, (GY - ax[1][0]) / h, (GZ - ax[2][0]) / h])
        dl = map_coordinates(d, loc, order=1, mode="nearest")
        sl = tuple(slice(a_, b_) for a_, b_ in zip(i0, i1))
        old = self.sdf[sl]
        # The global grid cannot hold the feather-thin edge of a low cap.  Offset the
        # liquid surface by a few microns (bisection) so that the volume it adds, as
        # measured on the global grid, is exactly the droplet volume.
        occ_old = np.clip(0.5 - old / self.res, 0, 1)
        lo_d, hi_d = -self.res, self.res
        for _ in range(12):
            mid = 0.5 * (lo_d + hi_d)
            add = float((np.clip(0.5 - np.minimum(old, dl - mid) / self.res, 0, 1) - occ_old).sum()) * self.res ** 3
            if add < volume:
                lo_d = mid
            else:
                hi_d = mid
        dl = dl - 0.5 * (lo_d + hi_d)
        newly = (dl < 0) & (old >= 0)
        np.minimum(old, np.clip(dl, -self.band, self.band).astype(np.float32), out=old)
        self.n_deposits += 1
        if getattr(self, "owner", None) is not None:
            self.owner[sl][newly] = self.n_deposits
        cen = np.array([X[liq].mean(), Y[liq].mean(), Z[liq].mean()]) if liq.any() else np.asarray(contact, float)
        return cen, r

    def surface_point(self, p, search=0.9e-3):
        """Closest point of the current surface to p, and the outward normal there.

        Gradient projection first; if p sits where the clipped field is flat
        (buried, or too far out) fall back to a brute-force search of the
        neighbourhood.  Returns (contact, normal, ok)."""
        c = self.project(np.asarray(p, float)[None], iters=6)[0]
        if np.all(np.isfinite(c)) and abs(self.dist(c)[0]) < 0.25 * self.res:
            n = self.normal(c[None])[0]
            if np.isfinite(n).all() and np.linalg.norm(n) > 0.5:
                return c, n, True
        i0, i1 = self._box(p - search, p + search)
        sl = tuple(slice(a, b) for a, b in zip(i0, i1))
        loc = self.sdf[sl]
        m = np.abs(loc) < self.res
        if not m.any():
            return c, np.array([0.0, 0.0, -1.0]), False
        idx = np.argwhere(m)
        pts = self.lo + (idx + i0) * self.res
        c = pts[np.argmin(np.linalg.norm(pts - p, axis=1))]
        c = self.project(c[None], iters=4)[0]
        n = self.normal(c[None])[0]
        ok = bool(np.isfinite(n).all() and np.linalg.norm(n) > 0.5 and np.isfinite(c).all())
        return c, (n if ok else np.array([0.0, 0.0, -1.0])), ok

    def approach_clear(self, contact, direction, a, reach=18e-3, margin=0.12e-3, skip=0.45e-3):
        """Is the straight approach along `direction` (pointing away from the part)
        free of collisions for a droplet of radius a?"""
        s = np.arange(skip, reach, 0.12e-3)
        pts = contact + direction * (a + s)[:, None]
        inside = np.all((pts >= self.lo) & (pts <= self.lo + self.res * (np.array(self.shape) - 1)), axis=1)
        if not inside.any():
            return True
        return bool(self.dist(pts[inside]).min() >= min(a + margin, self.band * 0.95))


def tilted_directions(n, tilt=np.radians(35), k=6):
    """Alternative approach directions on a cone around the normal."""
    fr = frame_from_normal(n)
    out = [n]
    for i in range(k):
        ph = 2 * np.pi * i / k
        out.append(np.cos(tilt) * n + np.sin(tilt) * (np.cos(ph) * fr[0] + np.sin(ph) * fr[1]))
    return out
