"""Study 10: can the sound field shape a droplet that has just touched down?

A wet 0.3 mm droplet (45 deg cap) sits on the underside of the 20 mm test part.
The time-averaged acoustic radiation pressure on its (momentarily rigid) surface is
    P_rad = |p|^2 / (4 rho0 c0^2)  -  rho0 |v|^2 / 4 ,   v = grad p / (i w rho0).
Only *differences* of P_rad over the droplet can change its shape.  We compute
dP = P_rad(crest) - P_rad(rim) for (a) the delivery funnel, and for holograms
optimised to (b) maximise it (squash the cap flat) and (c) minimise it (pull the
cap tall), all at the full 20 Vpp drive, and compare with the capillary pressure
that resists reshaping.
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replicator import constants as C, configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field, frame_from_normal
from replicator.holo import TrapSolver
from replicator.printsim import cap_sphere_radius

R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
th = np.radians(45)
rs = cap_sphere_radius(drop.a, th)
hcap, foot = rs * (1 - np.cos(th)), rs * np.sin(th)
for sname, nvec in (("underside", [0, 0, -1.0]), ("side wall", [1.0, 0, 0])):
    n = np.array(nvec)
    part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.1e-3)
    part.add_sphere([0, 0, 0], R)
    xs = n * R
    part.add_sphere(xs - n * rs * np.cos(th), rs)                 # the wet droplet, as a cap on the surface
    fld = build_field(part, arr, target=xs + n * hcap,
                      levels=[(3.5e-3, 0.8e-3), (1.8e-3, 0.4e-3), (0.9e-3, 0.2e-3), (0.6e-3, 0.09e-3)])
    fr = frame_from_normal(n)
    off = 0.07e-3
    crest = (xs + n * (hcap + off))[None]
    # rim: points on the cap at 85 % of the footprint radius, lifted off along the cap normal
    rim = []
    c_s = xs - n * rs * np.cos(th)
    for ph in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        rho = 0.85 * foot
        t = np.cos(ph) * fr[0] + np.sin(ph) * fr[1]
        p_on = c_s + rho * t + n * np.sqrt(rs ** 2 - rho ** 2)
        nn = (p_on - c_s) / rs
        rim.append(p_on + nn * off)
    pts = np.vstack([crest, np.array(rim)])
    A, G = fld.rows(pts)
    k1 = 1.0 / (4 * C.RHO0 * C.C0 ** 2); k2 = 1.0 / (4 * C.RHO0 * C.OMEGA ** 2)
    wts = np.concatenate([[1.0], -np.ones(8) / 8])

    def prad(w):
        return k1 * np.abs(A @ w) ** 2 - k2 * np.sum(np.abs(G @ w) ** 2, axis=1)

    def obj(phi, sign):
        w = np.exp(1j * phi)
        P = A @ w; Gv = G @ w
        val = wts @ (k1 * np.abs(P) ** 2 - k2 * np.sum(np.abs(Gv) ** 2, axis=1))
        q = k1 * (np.conj(P) * wts)[:, None] * A - k2 * np.einsum("sd,sdt->st", np.conj(Gv) * wts[:, None], G)
        return -sign * val, -sign * (-2.0 * np.imag(w * q.sum(0)))

    sv = TrapSolver(fld, drop)
    phi_f, s_f, info = sv.solve_funnel(xs, fr, np.array([0.03, 0.3, 0.6, 0.9, 1.2]) * 1e-3 + hcap, n_random=1)
    pf = prad(s_f * np.exp(1j * phi_f))
    best = {}
    for sign, nm in ((+1, "squash"), (-1, "pull tall")):
        b = None
        for seed in range(3):
            x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, arr.n) if seed else arr.focus_phases(crest[0])
            r = minimize(obj, x0, args=(sign,), jac=True, method="L-BFGS-B", options=dict(maxiter=400))
            if b is None or r.fun < b.fun:
                b = r
        best[nm] = prad(np.exp(1j * b.x))
    p_cap = 2 * drop.sigma / rs
    print(f"== {sname}: wet cap height {hcap*1e6:.0f} um, footprint {2*foot*1e6:.0f} um; capillary pressure 2 sigma/R = {p_cap:.0f} Pa")
    print(f"   delivery funnel : crest {pf[0]:+7.1f} Pa, rim {pf[1:].mean():+7.1f} Pa -> crest-minus-rim {pf[0]-pf[1:].mean():+7.1f} Pa "
          f"({(pf[0]-pf[1:].mean())/p_cap*100:+.0f} % of capillary)")
    for nm, pp in best.items():
        d = pp[0] - pp[1:].mean()
        print(f"   best '{nm:9s}': crest {pp[0]:+7.1f} Pa, rim {pp[1:].mean():+7.1f} Pa -> crest-minus-rim {d:+7.1f} Pa "
              f"({d/p_cap*100:+.0f} % of capillary)", flush=True)
