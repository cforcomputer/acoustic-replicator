"""Growing-part geometry: narrow-band signed-distance voxel model, BEM meshing,
local mesh refinement around the landing site."""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from .acoustics import Mesh


class VoxelPart:
    """Signed distance field on a regular grid (negative inside the solid).

    The solid is the union of primitives (pole capsule + cured droplets); the
    field is the running minimum of the primitive SDFs, clipped to +-band.
    """

    def __init__(self, lo, hi, res=0.1e-3, band=2.5e-3):
        self.lo = np.asarray(lo, float)
        self.res = float(res)
        self.shape = tuple(int(round(s)) + 1 for s in (np.asarray(hi) - self.lo) / res)
        self.band = band
        self.sdf = np.full(self.shape, band, np.float32)
        self.n_deposits = 0

    # -- coordinates ---------------------------------------------------------
    def axes(self):
        return [self.lo[d] + self.res * np.arange(self.shape[d]) for d in range(3)]

    def _box(self, lo, hi):
        i0 = np.maximum(np.floor((np.asarray(lo) - self.lo) / self.res).astype(int), 0)
        i1 = np.minimum(np.ceil((np.asarray(hi) - self.lo) / self.res).astype(int) + 1, self.shape)
        return i0, i1

    def _local_grid(self, i0, i1):
        ax = [self.lo[d] + self.res * np.arange(i0[d], i1[d]) for d in range(3)]
        return np.meshgrid(*ax, indexing="ij", sparse=True)

    # -- primitives ----------------------------------------------------------
    def _merge(self, i0, i1, d):
        sl = tuple(slice(a, b) for a, b in zip(i0, i1))
        np.minimum(self.sdf[sl], np.clip(d, -self.band, self.band).astype(np.float32), out=self.sdf[sl])

    def add_sphere(self, c, r):
        c = np.asarray(c, float)
        i0, i1 = self._box(c - r - self.band, c + r + self.band)
        X, Y, Z = self._local_grid(i0, i1)
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r
        self._merge(i0, i1, d)

    def add_capsule(self, a, b, r):
        a = np.asarray(a, float); b = np.asarray(b, float)
        i0, i1 = self._box(np.minimum(a, b) - r - self.band, np.maximum(a, b) + r + self.band)
        X, Y, Z = self._local_grid(i0, i1)
        ab = b - a
        t = ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / (ab @ ab)
        t = np.clip(t, 0, 1)
        d = np.sqrt((X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2
                    + (Z - a[2] - t * ab[2]) ** 2) - r
        self._merge(i0, i1, d)

    def add_cylinder(self, a, b, r):
        """Flat-ended cylinder from a to b (axis must be along z)."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        z0, z1 = min(a[2], b[2]), max(a[2], b[2])
        i0, i1 = self._box([a[0] - r - self.band, a[1] - r - self.band, z0 - self.band],
                           [a[0] + r + self.band, a[1] + r + self.band, z1 + self.band])
        X, Y, Z = self._local_grid(i0, i1)
        dr = np.sqrt((X - a[0]) ** 2 + (Y - a[1]) ** 2) - r
        dz = np.maximum(z0 - Z, Z - z1)
        out = np.sqrt(np.maximum(dr, 0) ** 2 + np.maximum(dz, 0) ** 2)
        d = np.where((dr < 0) & (dz < 0), np.maximum(dr, dz), out)
        self._merge(i0, i1, d)

    def volume(self):
        """Solid volume with sub-voxel accuracy (linear occupancy ramp across the surface)."""
        occ = np.clip(0.5 - self.sdf / self.res, 0, 1)
        return float(occ.sum()) * self.res ** 3

    # -- queries -------------------------------------------------------------
    def dist(self, pts):
        pts = np.atleast_2d(pts)
        idx = ((pts - self.lo) / self.res).T
        return map_coordinates(self.sdf, idx, order=1, mode="nearest")

    def normal(self, pts, h=None):
        h = h or self.res
        pts = np.atleast_2d(pts)
        g = np.zeros_like(pts)
        for d in range(3):
            e = np.zeros(3); e[d] = h
            g[:, d] = self.dist(pts + e) - self.dist(pts - e)
        n = np.linalg.norm(g, axis=1)
        return g / np.where(n > 0, n, 1)[:, None]

    def project(self, pts, iters=4):
        """Project points onto the surface (sdf = 0) along the SDF gradient."""
        pts = np.array(np.atleast_2d(pts), float)
        for _ in range(iters):
            d = self.dist(pts)
            pts -= d[:, None] * self.normal(pts)
        return pts

    # -- meshing -------------------------------------------------------------
    def to_mesh(self, h=1.4e-3, zmax=None, mc=0.6e-3):
        """Surface mesh for the BEM with edge length ~h.

        Marching cubes at spacing `mc`, quadric decimation to the target panel
        count, then bisection of any remaining long edges (decimation leaves
        slender panels on the pole).  zmax truncates the pole.
        """
        import fast_simplification as fs
        s = max(1, int(round(mc / self.res)))
        f = self.sdf[::s, ::s, ::s].astype(np.float64)
        ax = [a[::s] for a in self.axes()]
        if zmax is not None:
            f = np.maximum(f, (ax[2] - zmax)[None, None, :])
        f = np.pad(f, 1, mode="constant", constant_values=self.band)
        v, faces, _, _ = marching_cubes(f, level=0.0)
        v = (v - 1.0) * (self.res * s) + self.lo
        m = Mesh(v, faces)
        target = int(m.area.sum() / (0.433 * h ** 2))
        if target < m.n:
            v2, f2 = fs.simplify(m.verts, m.faces, target_count=target, agg=5)
            m = Mesh(v2, f2)
        return split_long(m, 1.45 * h)


def split_long(mesh: Mesh, max_edge):
    """Bisect panels across their longest edge until no edge exceeds max_edge."""
    tri = mesh.tri.copy()
    for _ in range(12):
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], 1)
        need = e.max(1) > max_edge
        if not need.any():
            break
        t = tri[need]
        k = e[need].argmax(1)                       # longest edge k: vertices k, k+1
        ar = np.arange(len(t))
        a = t[ar, k]; b = t[ar, (k + 1) % 3]; c = t[ar, (k + 2) % 3]
        m = 0.5 * (a + b)
        tri = np.concatenate([tri[~need], np.stack([a, m, c], 1), np.stack([m, b, c], 1)])
    verts = tri.reshape(-1, 3)
    return Mesh(verts, np.arange(len(verts)).reshape(-1, 3))


def refine_near(mesh: Mesh, target, levels, project=None):
    """Subdivide (1 -> 4) panels near `target` until they meet the size limits.

    levels: [(radius, max_edge), ...]  e.g. [(3e-3, 0.5e-3), (1.2e-3, 0.2e-3)]
    project: optional callable mapping points onto the true surface.
    The result is non-conforming, which is fine for constant collocation panels.
    """
    target = np.asarray(target, float)
    tri = mesh.tri.copy()
    for _ in range(8):
        cen = tri.mean(1)
        e = np.maximum.reduce([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                               np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                               np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)])
        dist = np.linalg.norm(cen - target, axis=1)
        need = np.zeros(len(tri), bool)
        for (rad, hmax) in levels:
            need |= (dist < rad + 0.5 * e) & (e > hmax)
        if not need.any():
            break
        t = tri[need]
        m01 = 0.5 * (t[:, 0] + t[:, 1]); m12 = 0.5 * (t[:, 1] + t[:, 2]); m20 = 0.5 * (t[:, 2] + t[:, 0])
        if project is not None:
            m01 = project(m01); m12 = project(m12); m20 = project(m20)
        new = np.concatenate([
            np.stack([t[:, 0], m01, m20], 1), np.stack([m01, t[:, 1], m12], 1),
            np.stack([m20, m12, t[:, 2]], 1), np.stack([m01, m12, m20], 1)])
        tri = np.concatenate([tri[~need], new])
    verts = tri.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    return Mesh(verts, faces)
