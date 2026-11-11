"""Validation of the acoustic layer against closed-form results.

V1  piston sub-source model  vs  2*J1(ka sin t)/(ka sin t) directivity
V2  analytic gradients       vs  finite differences
V3  BEM (hard sphere + point source)  vs  exact spherical-harmonic series,
    on the surface, close to the surface (down to 0.1 mm) and far away,
    including a fictitious-resonance frequency with and without CHIEF.
"""
import sys, time
import numpy as np
from scipy.special import j1, spherical_jn, spherical_yn, eval_legendre
import trimesh

sys.path.insert(0, ".")
from replicator import constants as C
from replicator.acoustics import Array, Panel, Mesh, Scatterer, Field
from replicator.kernels import dl_matrices

k = C.K
ok_all = True


def check(name, val, tol):
    global ok_all
    ok = val < tol
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e})")


# ---------------------------------------------------------------- V1
print("V1 piston directivity")
pan = Panel("one", [0, 0, 0], [0, 0, 1], [1, 0, 0], 1, 1)
arr = Array([pan], vpp=1.0, p_ref_per_vpp=1.0)
th = np.radians(np.linspace(0, 70, 71))
d = 0.5
pts = np.stack([d * np.sin(th), 0 * th, d * np.cos(th)], 1)
A, _ = arr.rows(pts, grad=False)
num = np.abs(A[:, 0]) * d
x = k * C.PISTON_RADIUS * np.sin(th)
ana = np.where(x > 1e-9, 2 * j1(x) / np.where(x > 1e-9, x, 1), 1.0)
m50 = th <= np.radians(50)
check("max |D_num - D_bessel| for theta<=50deg", np.abs(num - ana)[m50].max(), 2e-2)
check("max |D_num - D_bessel| for theta<=70deg", np.abs(num - ana).max(), 5e-2)
print(f"     D(40deg) = {num[40]:.3f}  (datasheet -6 dB full angle 80deg -> 0.50)")

# ---------------------------------------------------------------- V2
print("V2 gradients vs finite differences")
rng = np.random.default_rng(1)
p0 = np.array([[0.01, -0.02, 0.07]])
h = 1e-6
A0, G0 = arr.rows(p0)
for dd in range(3):
    e = np.zeros(3); e[dd] = h
    Ap, _ = arr.rows(p0 + e, grad=False); Am, _ = arr.rows(p0 - e, grad=False)
    fd = (Ap - Am) / (2 * h)
    check(f"piston grad[{dd}] rel err", abs(fd[0, 0] - G0[0, dd, 0]) / abs(G0[0, :, 0]).max(), 1e-6)

sph = trimesh.creation.icosphere(subdivisions=2, radius=5e-3)
msh = Mesh(sph.vertices, sph.faces)
for dist in (3e-3, 0.3e-3):                       # far-ish and very near the surface
    x0 = np.array([[0.3, 0.5, 0.81]]); x0 = x0 / np.linalg.norm(x0) * (5e-3 + dist)
    none = np.array([-1], np.int64)
    Dp, Dg = dl_matrices(x0, msh.tri, msh.nrm, msh.area, msh.size, k, 4.0, none, True)
    mu = rng.normal(size=msh.n) + 1j * rng.normal(size=msh.n)
    g = Dg[0] @ mu
    fd = np.zeros(3, complex)
    hh = 1e-7
    for dd in range(3):
        e = np.zeros(3); e[dd] = hh
        Dpp, _ = dl_matrices(x0 + e, msh.tri, msh.nrm, msh.area, msh.size, k, 4.0, none, False)
        Dpm, _ = dl_matrices(x0 - e, msh.tri, msh.nrm, msh.area, msh.size, k, 4.0, none, False)
        fd[dd] = ((Dpp - Dpm)[0] @ mu) / (2 * hh)
    check(f"double-layer grad rel err at {dist*1e3:.1f} mm", np.abs(fd - g).max() / np.abs(g).max(), 1e-5)

# closed surface identity: sum_j D_ij = -1 inside, 0 outside (k -> 0)
xin = np.array([[1e-3, 0.5e-3, -1e-3]]); xout = np.array([[9e-3, 0, 0]])
for nm, xx, expect in (("inside", xin, -1.0), ("outside", xout, 0.0)):
    Dp, _ = dl_matrices(xx, msh.tri, msh.nrm, msh.area, msh.size, 1e-9, 1e9,
                        np.array([-1], np.int64), False)
    check(f"static double-layer sum {nm}", abs(Dp.sum() - expect), 1e-10)


# ---------------------------------------------------------------- V3
def sphere_series(a, rs, pts, nmax=70):
    """Total field of exp(ikR)/R point source at (0,0,rs) outside a hard sphere radius a."""
    r = np.linalg.norm(pts, axis=1)
    cg = pts[:, 2] / r
    R = np.linalg.norm(pts - np.array([0, 0, rs]), axis=1)
    p = np.exp(1j * k * R) / R
    for n in range(nmax):
        hn_rs = spherical_jn(n, k * rs) + 1j * spherical_yn(n, k * rs)
        dj = spherical_jn(n, k * a, True)
        dh = dj + 1j * spherical_yn(n, k * a, True)
        hn_r = spherical_jn(n, k * r) + 1j * spherical_yn(n, k * r)
        term = -1j * k * (2 * n + 1) * hn_rs * (dj / dh) * hn_r * eval_legendre(n, cg)
        if not np.all(np.isfinite(term)):
            break
        p = p + term
    return p


def series_grad(a, rs, pts, h=2e-7):
    g = np.zeros((len(pts), 3), complex)
    for dd in range(3):
        e = np.zeros(3); e[dd] = h
        g[:, dd] = (sphere_series(a, rs, pts + e) - sphere_series(a, rs, pts - e)) / (2 * h)
    return g


def run_sphere(a, rs, subdiv, n_chief, label):
    sph = trimesh.creation.icosphere(subdivisions=subdiv, radius=a)
    # push vertices so the flat-panel polyhedron has the same volume as the sphere
    msh = Mesh(sph.vertices, sph.faces)
    scale = (4 / 3 * np.pi * a ** 3 / msh.signed_volume()) ** (1 / 3)
    msh = Mesh(sph.vertices * scale, sph.faces)
    src = Array([Panel("src", [0, 0, rs], [0, 0, -1], [1, 0, 0], 1, 1)], vpp=1.0,
                p_ref_per_vpp=1.0, piston_radius=1e-7)
    t0 = time.time()
    sc = Scatterer(msh, src, n_chief=n_chief)
    fld = Field(src, sc)
    w = np.ones(1, complex)
    mu = sc.M @ w
    ex = sphere_series(a, rs, msh.cen / np.linalg.norm(msh.cen, axis=1)[:, None] * a)
    e_surf = np.abs(mu - ex).max() / np.abs(ex).max()
    print(f"  {label}: {msh.n} panels (edge ~{msh.size.mean()*1e3:.2f} mm), ka={k*a:.2f}, "
          f"solve {time.time()-t0:.1f}s")
    print(f"     surface pressure   max rel err {e_surf:.3e}")
    out = {"surf": e_surf}
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(400, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    for dist in (0.1e-3, 0.2e-3, 0.5e-3, 1e-3, 2.1e-3, 5e-3, 20e-3):
        pts = dirs * (a + dist)
        p, g = fld.eval(pts, w)
        pe = sphere_series(a, rs, pts)
        ge = series_grad(a, rs, pts)
        scale_p = np.abs(pe).max()
        scale_g = np.linalg.norm(ge, axis=1).max()
        ep = np.abs(p - pe) / scale_p
        eg = np.linalg.norm(g - ge, axis=1) / scale_g
        print(f"     d={dist*1e3:5.1f} mm   p err: rms {np.sqrt((ep**2).mean()):.3e} max {ep.max():.3e}"
              f"   grad err: rms {np.sqrt((eg**2).mean()):.3e} max {eg.max():.3e}")
        out[dist] = (np.sqrt((ep ** 2).mean()), ep.max(), np.sqrt((eg ** 2).mean()), eg.max())
    return out


print("V3 BEM vs exact series (hard sphere, point source at 60 mm)")
r = run_sphere(10e-3, 60e-3, 3, 40, "sphere R=10mm, coarse")
check("coarse mesh: surface pressure", r["surf"], 5e-2)
check("coarse mesh: p rms err at 2.1 mm", r[2.1e-3][0], 3e-2)
r = run_sphere(10e-3, 60e-3, 4, 40, "sphere R=10mm, fine")
check("fine mesh: surface pressure", r["surf"], 2e-2)
check("fine mesh: p rms err at 0.2 mm", r[0.2e-3][0], 2e-2)
check("fine mesh: grad rms err at 0.5 mm", r[0.5e-3][2], 5e-2)

print("V3b fictitious interior resonance (ka = 2*pi, Dirichlet eigenvalue of the sphere)")
a_res = 2 * np.pi / k
r0 = run_sphere(a_res, 60e-3, 3, 0, "no CHIEF")
r1 = run_sphere(a_res, 60e-3, 3, 40, "with CHIEF")
check("CHIEF: p rms err at 2.1 mm at resonance", r1[2.1e-3][0], 3e-2)
print(f"     (without CHIEF the same error is {r0[2.1e-3][0]:.3e})")

print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
