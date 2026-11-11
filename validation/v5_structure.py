"""Checks of the structural model (replicator/structure.py).

S1  A thin cantilever plate, two elements thick, against beam theory (tip load and own weight).
    Also shows why plain trilinear cubes cannot be used (far too stiff).
S2  Radiation force on a small rigid sphere: surface integral of the radiation pressure computed
    from the boundary-element surface pressure, against Gor'kov's formula.
"""
import sys
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import structure as S, configs, constants as C
from replicator.acoustics import Mesh, Scatterer, Field

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


print("S1 cantilever plate vs beam theory")
h, E, nu, rho = 0.18e-3, 1e9, 0.35, 1150.0
for nt in (2, 3):
    L, b, t = 33 * h, 6 * h, nt * h
    nx, ny, nz = 33 + 4, 6, nt
    occ = np.ones((nx, ny, nz))
    fixed = lambda p: p[:, 0] <= 4 * h + 1e-9
    I = b * t ** 3 / 12
    for inc in (True, False):
        m = S.Model(occ, [0, 0, 0], h, fixed, E=E, nu=nu, rho=rho, incompatible=inc)
        tip = np.array([4 * h + L, b / 2, t / 2])
        F = 1e-6
        f = np.zeros((m.nn, 3))
        end = np.flatnonzero(np.abs(m.xyz[:, 0] - (4 * h + L)) < 1e-9)
        f[end, 2] = -F / len(end)
        u = m.solve(f)
        d_fe = -m.at(u, tip)[0, 2]
        d_th = F * L ** 3 / (3 * E * I)
        ug = m.solve(m.gravity())
        dg_fe = -m.at(ug, tip)[0, 2]
        dg_th = rho * S.G_ACC * b * t * L ** 4 / (8 * E * I)
        if inc:
            # a wide plate is up to 1/(1-nu^2) = 1.14x stiffer than a beam; accept 0.85-1.05 of beam theory
            check(f"{nt} elements thick, incompatible modes: tip load", 0.85 < d_fe / d_th < 1.05, f"FE {d_fe*1e6:.3f} um, beam {d_th*1e6:.3f} um, ratio {d_fe/d_th:.3f}")
            check(f"{nt} elements thick, incompatible modes: own weight", 0.85 < dg_fe / dg_th < 1.05, f"FE {dg_fe*1e9:.2f} nm, beam {dg_th*1e9:.2f} nm, ratio {dg_fe/dg_th:.3f}")
        else:
            print(f"       (plain trilinear cubes, {nt} thick: ratio {d_fe/d_th:.3f} - too stiff, not used)")

print("S2 radiation force on a small rigid sphere vs Gor'kov")
import trimesh
arr = configs.REGISTRY["rco5"](18.0)
a = 0.3e-3
x_focus = np.array([0.0, 0.0, 0.0])
phi = arr.focus_phases(x_focus)
w = np.exp(1j * phi)
k = arr.k
rho0, c0 = C.RHO0, C.C0
vol = 4 / 3 * np.pi * a ** 3
fld0 = Field(arr)


def gorkov_U(x):
    p, g = fld0.eval(np.atleast_2d(x), w)
    v2 = (np.abs(g) ** 2).sum(1) / (rho0 * k * c0) ** 2
    return vol * (np.abs(p) ** 2 / (4 * rho0 * c0 ** 2) - 0.75 * rho0 * v2 / 2)        # rigid, fixed sphere: f1 = f2 = 1


for off in (np.array([0.0, 0.0, 0.9e-3]), np.array([0.7e-3, 0.0, 0.5e-3])):
    ico = trimesh.creation.icosphere(subdivisions=3, radius=a)
    # scale the faceted sphere so that it has the volume of the true one
    sc = (vol / abs(ico.volume)) ** (1 / 3)
    mesh = Mesh(ico.vertices * sc + off, ico.faces)
    scat = Scatterer(mesh, arr, n_chief=8)
    ps = scat.M @ w
    P, F, _ = S.radiation_pressure(mesh, ps, k)
    F_surf = F.sum(0)
    hh = 0.02e-3
    F_g = -np.array([(gorkov_U(off + hh * e) - gorkov_U(off - hh * e))[0] / (2 * hh) for e in np.eye(3)])
    err = np.linalg.norm(F_surf - F_g) / np.linalg.norm(F_g)
    check(f"sphere at {np.round(off*1e3,2)} mm", err < 0.15, f"surface integral {np.round(F_surf*1e9,2)} nN, Gor'kov {np.round(F_g*1e9,2)} nN, difference {100*err:.1f} % (ka = {k*a:.2f})")

print("\nALL PASS" if ok_all else "\nFAILURES")
