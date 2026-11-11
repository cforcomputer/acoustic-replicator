"""Linear-elastic check of the half-built part: can it carry itself and the sound field?

The cured part hangs from the build pin.  While it is printed it is loaded by its
own weight and by the acoustic radiation pressure of the field that brings the
next droplet; thin walls that hang free are the weak spots.  This module turns a
snapshot of the print (voxel signed-distance field) into a finite-element model:

  * 8-node cube elements, one per voxel, with Wilson-Taylor incompatible modes so
    that a wall only two elements thick still bends correctly (plain trilinear
    cubes are far too stiff in bending);
  * partial voxels carry a proportionally reduced stiffness and weight;
  * everything inside the steel pin is fixed;
  * the system is solved with smoothed-aggregation multigrid + conjugate gradients.

Displacements scale as 1/E, so one solve at E = 1 GPa covers every modulus.
Units: SI.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

G_ACC = 9.81


def _D(nu):
    lam = nu / ((1 + nu) * (1 - 2 * nu)); mu = 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = mu * np.eye(3)
    return D                                           # for E = 1; strain order xx yy zz xy yz zx (engineering shear)


_CORN = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])       # node order inside an element
_SGN = 2.0 * _CORN - 1.0


def _B(xi):
    """Strain-displacement matrices at natural point xi for a unit cube: nodal (6,24) and incompatible-mode (6,9)."""
    dN = np.zeros((8, 3))
    for a in range(8):
        s = _SGN[a]
        f = 1 + s * xi
        dN[a] = 2 * np.array([s[0] * f[1] * f[2], f[0] * s[1] * f[2], f[0] * f[1] * s[2]]) / 8.0     # d/dx = 2 d/dxi
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = dN[a]
        B[:, 3 * a:3 * a + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz], [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
    Gm = np.zeros((6, 9))
    for m in range(3):                                 # bubble 1 - xi_m^2 added to each displacement component
        d = np.zeros(3); d[m] = -4.0 * xi[m]
        dx, dy, dz = d
        Gm[:, 3 * m:3 * m + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz], [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
    return B, Gm


def element(nu=0.35, incompatible=True):
    """Stiffness of a unit cube with E = 1 (24x24), centroid strain matrix, and D."""
    D = _D(nu)
    g = 1 / np.sqrt(3)
    Kuu = np.zeros((24, 24)); Kua = np.zeros((24, 9)); Kaa = np.zeros((9, 9))
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                B, Gm = _B(np.array([a, b, c]))
                Kuu += B.T @ D @ B / 8.0; Kua += B.T @ D @ Gm / 8.0; Kaa += Gm.T @ D @ Gm / 8.0
    K = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T) if incompatible else Kuu
    B0, _ = _B(np.zeros(3))
    return 0.5 * (K + K.T), B0, D


class Model:
    """Finite-element model of a voxel occupancy field.

    occ   : (nx, ny, nz) fraction of each voxel that is solid
    lo, h : position of voxel (0,0,0)'s corner and the voxel size
    fixed : function(points) -> bool, True for nodes held by the pin
    """

    def __init__(self, occ, lo, h, fixed, E=1e9, nu=0.35, rho=1150.0, thr=0.35, incompatible=True, verbose=False):
        self.h, self.lo, self.E, self.nu, self.rho = float(h), np.asarray(lo, float), float(E), nu, rho
        act = occ >= thr
        # keep only what is connected to the fixed region (6-connectivity)
        lab, n = ndi.label(act)
        cen = lambda idx: self.lo + (np.asarray(idx, float) + 0.5) * self.h
        ijk = np.argwhere(act)
        held = fixed(cen(ijk))
        keep = np.unique(lab[tuple(ijk[held].T)])
        act &= np.isin(lab, keep)
        self.dropped = int((occ >= thr).sum() - act.sum())
        self.ijk = np.argwhere(act)
        self.occ = occ[tuple(self.ijk.T)].astype(float)
        ne = len(self.ijk)
        nx, ny, nz = occ.shape
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        en = np.stack([nid(self.ijk[:, 0] + c[0], self.ijk[:, 1] + c[1], self.ijk[:, 2] + c[2]) for c in _CORN], 1)     # (ne, 8) global node ids
        uniq, inv = np.unique(en, return_inverse=True)
        self.en = inv.reshape(en.shape)
        self.nn = len(uniq)
        k = uniq % (nz + 1); j = (uniq // (nz + 1)) % (ny + 1); i = uniq // ((nz + 1) * (ny + 1))
        self.xyz = self.lo + np.stack([i, j, k], 1) * self.h
        self.fix = fixed(self.xyz)
        Ke, self.B0, self.D = element(nu, incompatible)
        edof = (3 * self.en[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 24)
        scale = self.E * self.h * self.occ
        rows = np.repeat(edof, 24, axis=1).ravel()
        cols = np.tile(edof, (1, 24)).ravel()
        data = (scale[:, None] * Ke.ravel()[None, :]).ravel()
        K = sp.coo_matrix((data, (rows, cols)), shape=(3 * self.nn, 3 * self.nn)).tocsr()
        del rows, cols, data
        self.edof = edof
        free = np.repeat(~self.fix, 3)
        self.free = np.flatnonzero(free)
        self.K = K[self.free][:, self.free].tocsr()
        self.tree = cKDTree(self.xyz)
        self._ml = None
        # surface nodes: nodes that belong to fewer than 8 elements
        cnt = np.bincount(self.en.ravel(), minlength=self.nn)
        self.surface = cnt < 8
        if verbose:
            print(f"  FE model: {ne} elements, {self.nn} nodes, {len(self.free)} free dof, {int(self.fix.sum())} fixed nodes, dropped {self.dropped} loose voxels")

    # ---------------------------------------------------------------- loads
    def gravity(self, g=(0, 0, -G_ACC)):
        f = np.zeros((self.nn, 3))
        w = self.rho * self.h ** 3 * self.occ / 8.0
        for a in range(8):
            np.add.at(f, self.en[:, a], w[:, None] * np.asarray(g, float)[None, :])
        return f

    def patch(self, x0, force, radius):
        """Force vector spread evenly over the surface nodes within `radius` of x0."""
        f = np.zeros((self.nn, 3))
        idx = np.array(self.tree.query_ball_point(np.asarray(x0, float), radius), dtype=int)
        idx = idx[self.surface[idx]] if len(idx) else idx
        if len(idx) == 0:
            idx = np.array([self.tree.query(np.asarray(x0, float))[1]])
        f[idx] = np.asarray(force, float) / len(idx)
        return f

    def panel_loads(self, centres, forces):
        """Forces given at arbitrary surface points (e.g. boundary-element panels) -> nearest surface node."""
        f = np.zeros((self.nn, 3))
        s_idx = np.flatnonzero(self.surface)
        _, j = cKDTree(self.xyz[s_idx]).query(centres)
        np.add.at(f, s_idx[j], forces)
        return f

    # ---------------------------------------------------------------- solve
    def solve(self, f, tol=1e-8, maxiter=400):
        import pyamg
        if self._ml is None:
            x = self.xyz[~self.fix]
            n = len(x)
            Bn = np.zeros((3 * n, 6))                      # rigid-body modes for the coarse spaces
            Bn[0::3, 0] = 1; Bn[1::3, 1] = 1; Bn[2::3, 2] = 1
            xc = (x - x.mean(0)) / self.h
            Bn[1::3, 3] = -xc[:, 2]; Bn[2::3, 3] = xc[:, 1]
            Bn[0::3, 4] = xc[:, 2];  Bn[2::3, 4] = -xc[:, 0]
            Bn[0::3, 5] = -xc[:, 1]; Bn[1::3, 5] = xc[:, 0]
            A = sp.bsr_matrix(self.K, blocksize=(3, 3))
            self._ml = pyamg.smoothed_aggregation_solver(A, B=Bn, strength=("symmetric", {"theta": 0.02}), max_coarse=600,
                                                         smooth=("jacobi", {"omega": 4.0 / 3.0}), improve_candidates=None)
        res = []
        u_f = self._ml.solve(f.reshape(-1)[self.free], tol=tol, maxiter=maxiter, accel="cg", residuals=res)
        self.last_residual = float(res[-1] / max(res[0], 1e-300)); self.last_iters = len(res) - 1
        u = np.zeros(3 * self.nn)
        u[self.free] = u_f
        return u.reshape(self.nn, 3)

    def von_mises(self, u):
        ue = u.reshape(-1)[self.edof]                                  # (ne, 24)
        eps = ue @ self.B0.T / self.h
        sig = self.E * (eps @ self.D.T)                                # stress in the solid fraction
        s = sig
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        return vm

    def at(self, u, pts):
        """Displacement at the model nodes nearest to pts."""
        _, j = self.tree.query(np.atleast_2d(pts))
        return u[j]


def occupancy_from_sdf(sdf, lo, res, st=3, zmax=None):
    """Coarsen a signed-distance snapshot by `st` and return the solid fraction of each coarse voxel."""
    nx, ny, nz = (np.array(sdf.shape) // st) * st
    s = (sdf[:nx, :ny, :nz] < 0).astype(np.float32)
    occ = s.reshape(nx // st, st, ny // st, st, nz // st, st).mean(axis=(1, 3, 5))
    lo2 = np.asarray(lo, float) - 0.5 * res                               # sdf samples sit at voxel centres
    if zmax is not None:
        kmax = int(np.floor((zmax - lo2[2]) / (res * st)))
        occ = occ[:, :, :max(kmax, 1)]
    return occ, lo2, res * st


def radiation_pressure(mesh, p_surf, k, rho0=1.204, c0=343.0, reach=2.6, p_cap=None):
    """Time-averaged radiation pressure on a rigid surface from the complex surface pressure.

    P = |p|^2 / (4 rho c^2) - |grad_s p|^2 / (4 rho omega^2), acting along the inward normal.
    The surface gradient is a least-squares fit over neighbouring panels that face the same way (panels on
    the two sides of a thin wall are close in space but are not neighbours on the surface).
    p_cap: amplitude limit for the surface pressure.  Constant-panel boundary elements break down where two
    faces of a thin wall are closer than a panel; a few panels then carry absurd values.  They are clipped to
    p_cap (pass a multiple of the largest incident amplitude) and counted.
    Returns P (Pa, per panel), the force vector on each panel (N), and the number of clipped panels."""
    cen, nrm, size = mesh.cen, mesh.nrm, mesh.size
    n_clip = 0
    if p_cap is not None:
        mag = np.abs(p_surf)
        bad = mag > p_cap
        n_clip = int(bad.sum())
        p_surf = np.where(bad, p_surf / np.maximum(mag, 1e-30) * p_cap, p_surf)
    tree = cKDTree(cen)
    omega = k * c0
    g2 = np.zeros(mesh.n)
    nb = tree.query_ball_point(cen, reach * size)
    for i in range(mesh.n):
        j = np.array([q for q in nb[i] if q != i and nrm[q] @ nrm[i] > 0.5], dtype=int)
        if len(j) < 3:
            continue
        n = nrm[i]
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(ref, n); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
        d = cen[j] - cen[i]
        A = np.stack([d @ t1, d @ t2], 1)
        w = 1.0 / (np.linalg.norm(d, axis=1) + 1e-12)
        sol, *_ = np.linalg.lstsq(A * w[:, None], (p_surf[j] - p_surf[i]) * w, rcond=None)
        g2[i] = float(np.abs(sol[0]) ** 2 + np.abs(sol[1]) ** 2)
    if p_cap is not None:
        g2 = np.minimum(g2, (k * p_cap) ** 2)                 # a gradient steeper than k * p is not a sound field
    P = np.abs(p_surf) ** 2 / (4 * rho0 * c0 ** 2) - g2 / (4 * rho0 * omega ** 2)
    F = -(P * mesh.area)[:, None] * nrm
    return P, F, n_clip
