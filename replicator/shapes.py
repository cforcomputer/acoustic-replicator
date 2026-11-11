"""Target shapes: voxelise a watertight STL into a signed distance field."""
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt, map_coordinates


@njit(cache=True)
def _zparity(tri, lo, res, nx, ny, nz):
    """Parity fill along z: toggle every voxel above each surface crossing."""
    occ = np.zeros((nx, ny, nz), np.uint8)
    for t in range(tri.shape[0]):
        ax, ay, az = tri[t, 0]; bx, by, bz = tri[t, 1]; cx, cy, cz = tri[t, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-30:
            continue
        ix0 = max(int(np.floor((min(ax, bx, cx) - lo[0]) / res)), 0)
        ix1 = min(int(np.ceil((max(ax, bx, cx) - lo[0]) / res)), nx - 1)
        iy0 = max(int(np.floor((min(ay, by, cy) - lo[1]) / res)), 0)
        iy1 = min(int(np.ceil((max(ay, by, cy) - lo[1]) / res)), ny - 1)
        for ix in range(ix0, ix1 + 1):
            px = lo[0] + ix * res
            for iy in range(iy0, iy1 + 1):
                py = lo[1] + iy * res
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                l3 = 1.0 - l1 - l2
                if l1 < 0 or l2 < 0 or l3 < 0:
                    continue
                z = l1 * az + l2 * bz + l3 * cz
                kz = int(np.ceil((z - lo[2]) / res))
                if kz < 0:
                    kz = 0
                for k in range(kz, nz):
                    occ[ix, iy, k] ^= 1
    return occ


class TargetShape:
    """Signed distance field of the object to print (negative inside)."""

    def __init__(self, sdf, lo, res):
        self.sdf = sdf.astype(np.float32)
        self.lo = np.asarray(lo, float)
        self.res = res
        self.shape = sdf.shape

    @staticmethod
    def from_mesh(verts, faces, res, pad=1.0e-3):
        verts = np.asarray(verts, float)
        lo = verts.min(0) - pad + np.array([1.7e-7, 2.3e-7, 0.0])     # avoid grid-on-edge ties
        hi = verts.max(0) + pad
        n = np.ceil((hi - lo) / res).astype(int) + 1
        occ = _zparity(np.ascontiguousarray(verts[faces]), lo, res, int(n[0]), int(n[1]), int(n[2])).astype(bool)
        d_out = distance_transform_edt(~occ) * res
        d_in = distance_transform_edt(occ) * res
        sdf = np.where(occ, -(d_in - 0.5 * res), d_out - 0.5 * res)
        return TargetShape(sdf, lo, res)

    def dist(self, pts):
        pts = np.atleast_2d(pts)
        idx = ((pts - self.lo) / self.res).T
        return map_coordinates(self.sdf, idx, order=1, mode="nearest")

    def volume(self):
        return float(np.clip(0.5 - self.sdf / self.res, 0, 1).sum()) * self.res ** 3

    def bounds(self):
        ii = np.argwhere(self.sdf < 0)
        return self.lo + ii.min(0) * self.res, self.lo + ii.max(0) * self.res


def load_benchy(path, scale=0.4, res=0.06e-3, pole_tip=(0.0, 0.0, 10e-3)):
    """Load the 3DBenchy STL, scale it, and place it hanging from the pole:
    the top of the chimney (highest point) sits on the pole tip."""
    import trimesh
    m = trimesh.load(path, force="mesh", process=True)
    v = np.asarray(m.vertices, float) * 1e-3 * scale          # STL is in mm
    top = v[:, 2].max()
    ring = v[v[:, 2] > top - 0.2e-3 * scale / 0.4]
    c = np.array([ring[:, 0].mean(), ring[:, 1].mean(), top])
    v = v - c + np.asarray(pole_tip)
    info = dict(watertight=bool(m.is_watertight), mesh_volume=float(abs(m.volume)) * 1e-9 * scale ** 3,
                n_faces=len(m.faces), size=(v.max(0) - v.min(0)))
    return TargetShape.from_mesh(v, np.asarray(m.faces), res), v, np.asarray(m.faces), info
