"""Study 2: end-to-end delivery of one droplet onto a 20 mm part (ideal plant).

Transit in a 3-D trap from a parked position to a stand-off point, short hold,
then press onto the surface.  Reports landing error and timing.
Usage: s2_single_delivery.py [config] [site] [--d 0.3]
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from replicator import constants as C
from replicator import configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.dynamics import Planner, Plant, Curve, simulate, concat

args = sys.argv[1:]
cfg = next((a for a in args if a in configs.REGISTRY), "box6_12")
DROP_D = float(args[args.index("--d") + 1]) * 1e-3 if "--d" in args else 0.3e-3
R = 10e-3
SITES = {
    "bottom": np.array([0, 0, -1.0]),
    "lower45": np.array([1, 0, -1.0]) / np.sqrt(2),
    "side": np.array([1.0, 0, 0]),
    "upper45": np.array([1, 0, 1.0]) / np.sqrt(2),
}
site = next((a for a in args if a in SITES), "bottom")
n = SITES[site]

drop = Droplet(DROP_D)
arr = configs.REGISTRY[cfg]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
xs = part.project((n * R)[None])[0]
t0 = time.time()
fld = build_field(part, arr, target=xs)
print(f"{arr.name}, droplet {DROP_D*1e3:.2f} mm, site {site}: {fld.scat.mesh.n} panels, BEM {time.time()-t0:.1f}s")

gap0 = 1.2e-3
x_pre = xs + n * (drop.a + gap0)
x_start = np.array([22e-3, 0, -22e-3])
path = Curve.bezier(x_start, x_pre, dir_in=-n)

pl = Planner(fld, drop, verbose='-v' in args)
t0 = time.time()
k1 = pl.transit(path)
k2 = pl.hold(x_pre, 0.02, k1.phi[-1], k1.amp[-1])
k3 = pl.press(xs, n, gap0, push=1.0, phi0=k2.phi[-1], s0=k2.amp[-1])
keys = concat(concat(k1, k2), k3)
print(f"planned {len(keys.t)} keyframes in {time.time()-t0:.1f}s; path {path.L*1e3:.1f} mm, "
      f"transit {k1.t[-1]*1e3:.0f} ms, total {keys.t[-1]*1e3:.0f} ms")
dphi = np.angle(np.exp(1j * np.diff(keys.phi, axis=0)))
print(f"keyframe-to-keyframe phase step: rms {np.sqrt((dphi**2).mean()):.3f} rad, "
      f"worst keyframe rms {np.sqrt((dphi**2).mean(1)).max():.3f} rad")
kap = np.array([i["kappa"] for i in k1.info])
print(f"transit trap: min kappa {kap.min():.2f}, median weakest {np.median(kap[:,0]):.2f} W/mm; "
      f"max Ba {max(i['Ba'] for i in keys.info):.2f}; max |f_res| {max(np.linalg.norm(i['f_res']) for i in keys.info):.3f}")

# droplet starts at rest at the equilibrium of the first hologram
plant = Plant(fld)
x0 = x_start + k1.info[0]["offset"]
t0 = time.time()
res = simulate(plant, drop, x0, np.zeros(3), keys, part=part)
print(f"simulated {res['t']*1e3:.0f} ms of flight in {time.time()-t0:.1f}s -> {res['status']}")
tr = res["traj"]
dev = np.linalg.norm(tr[:, 1:4] - tr[:, 4:7], axis=1)
print(f"tracking error during flight: rms {np.sqrt((dev**2).mean())*1e6:.0f} um, max {dev.max()*1e6:.0f} um; "
      f"max Ba seen by droplet {res['ba_max']:.2f}")
if res["status"] == "landed":
    contact = part.project(res["x"][None])[0]
    err = contact - xs
    print(f"LANDED: contact error {np.linalg.norm(err)*1e6:.0f} um (vector um {np.round(err*1e6)}), "
          f"impact speed {np.linalg.norm(res['v'])*1e3:.0f} mm/s, "
          f"We = {drop.rho*np.linalg.norm(res['v'])**2*drop.d/drop.sigma:.3f}")
np.save(f"out/s2_traj_{cfg}_{site}.npy", tr)
