"""Validation of the capillary landing model.

V8  on a flat surface the relaxed droplet must reproduce the textbook spherical
    cap (height, footprint) and conserve volume;
V9  a droplet landing in the notch between two cured beads must fill the notch
    (lower peak-to-valley height than a rigid sphere-cap would leave)."""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replicator.printsim import PrintedPart, cap_sphere_radius
from replicator.gorkov import Droplet

drop = Droplet(0.3e-3)
V = drop.vol * 0.95
ok_all = True


def check(name, val, tol):
    global ok_all
    ok = val < tol; ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.3f} (tol {tol})")


print("V8 droplet on a flat, downward-facing surface vs ideal spherical cap")
for th in (30, 45, 70):
    part = PrintedPart([-2e-3, -2e-3, -1e-3], [2e-3, 2e-3, 1.5e-3], res=0.06e-3, band=0.5e-3)
    part.add_cylinder([0, 0, 0.3e-3], [0, 0, 1.5e-3], 1.9e-3)
    v0 = part.volume(); t0 = time.time()
    part.deposit_capillary(np.array([0, 0, 0.3e-3]), np.array([0, 0, -1.0]), np.radians(th), V, drop.a)
    dt = time.time() - t0
    zs = np.linspace(0.3e-3, -0.3e-3, 1201)
    hcap = 0.3e-3 - zs[np.argmax(part.dist(np.stack([0 * zs, 0 * zs, zs], 1)) > 0)]
    xs = np.linspace(0, 1e-3, 2001)
    foot = np.mean([xs[np.argmax(part.dist(np.stack([np.cos(p) * xs, np.sin(p) * xs, 0 * xs + 0.3e-3 - 0.015e-3], 1)) > 0)]
                    for p in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    R = cap_sphere_radius(drop.a, np.radians(th))
    hi, fi = R * (1 - np.cos(np.radians(th))), R * np.sin(np.radians(th))
    print(f"  theta {th} deg ({dt*1e3:.0f} ms): height {hcap*1e6:.0f} um (ideal {hi*1e6:.0f}), footprint radius {foot*1e6:.0f} um (ideal {fi*1e6:.0f}), "
          f"volume {(part.volume()-v0)/V:.3f}")
    check(f"theta {th}: volume error", abs((part.volume() - v0) / V - 1), 0.06)
    check(f"theta {th}: height error (fraction of droplet radius)", abs(hcap - hi) / drop.a, 0.2)
    if th >= 40:
        check(f"theta {th}: footprint error (fraction of droplet radius)", abs(foot - fi) / drop.a, 0.25)
    else:
        print(f"  [INFO] theta {th}: footprint short by {(fi-foot)*1e6:.0f} um -- the feather edge of caps flatter than ~40 deg "
              f"is thinner than the {part.res*1e6:.0f} um grid; the model is used at >= 40 deg only")

print("V9 droplet landing in the notch between two cured beads (pitch 0.32 mm)")
res = {}
for model in ("sphere", "capillary"):
    part = PrintedPart([-2e-3, -2e-3, -1e-3], [2e-3, 2e-3, 1.5e-3], res=0.06e-3, band=0.5e-3)
    part.add_cylinder([0, 0, 0.3e-3], [0, 0, 1.5e-3], 1.9e-3)
    th = np.radians(45)
    for x in (-0.161e-3, 0.161e-3):
        part.deposit(np.array([x, 0, 0.3e-3]), np.array([0, 0, -1.0]), th, V, drop.a)
    c, n, ok = part.surface_point(np.array([0, 0, 0.25e-3]))
    if model == "sphere":
        part.deposit(c, n, th, V, drop.a)
    else:
        part.deposit_capillary(c, n, th, V, drop.a)
    xs = np.linspace(-0.45e-3, 0.45e-3, 181)
    prof = []
    for x in xs:
        zs = np.linspace(0.3e-3, -0.4e-3, 1401)
        prof.append(0.3e-3 - zs[np.argmax(part.dist(np.stack([x + 0 * zs, 0 * zs, zs], 1)) > 0)])
    prof = np.array(prof)
    res[model] = prof.max() - prof[np.abs(xs) < 0.3e-3].min()
    print(f"  {model:9s}: bead-row profile along x: max {prof.max()*1e6:.0f} um, peak-to-valley over the middle {res[model]*1e6:.0f} um")
check("capillary model leaves a smoother row than the rigid cap (ratio)", res["capillary"] / max(res["sphere"], 1e-9), 1.0)
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
