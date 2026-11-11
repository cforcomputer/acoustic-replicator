"""How much of the landing error is the simulation's own hologram keyframe spacing?"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replicator import configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.dynamics import Planner, Plant, simulate
from replicator.delivery import Site, Jet, plan_delivery

R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
sname = sys.argv[1] if len(sys.argv) > 1 else "upper45"
DIRS = {"bottom": [0, 0, -1], "side": [1, 0, 0], "upper45": [1, 0, 1]}
n = np.array(DIRS[sname], float); n /= np.linalg.norm(n)
xs = part.project((n * R)[None])[0]
site = Site(xs, part.normal(xs[None])[0])
fld = build_field(part, arr, target=xs)
jet = Jet()
t, x, v, a = jet.nominal(drop)
for ds in (0.3e-3, 0.15e-3):
    pl = Planner(fld, drop, ds=ds)
    t0 = time.time()
    keys, summ = plan_delivery(pl, jet, site, part, np.zeros(3), R)
    r = simulate(Plant(fld), drop, x[0], v[0], keys, part=part, record_every=2)
    tr = r["traj"]; dev = np.linalg.norm(tr[:, 1:4] - tr[:, 4:7], axis=1)
    c = part.project(r["x"][None])[0]
    print(f"{sname} ds={ds*1e3:.2f} mm: {summ['n_keys']} keyframes, plan {time.time()-t0:.0f}s, {r['status']}, "
          f"tracking rms {np.sqrt((dev**2).mean())*1e6:.0f} um max {dev.max()*1e6:.0f} um, landing error {np.linalg.norm(c-xs)*1e6:.0f} um", flush=True)
