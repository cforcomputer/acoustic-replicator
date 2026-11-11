import sys, time
import numpy as np
sys.path.insert(0, ".")
from replicator import constants as C, configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.dynamics import Planner, Plant, simulate
from replicator.delivery import Site, Jet, plan_delivery
import pickle, os

R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
DIRS = {"bottom": [0, 0, -1], "lower45": [1, 0, -1], "side": [1, 0, 0], "farside": [-1, 0, 0], "upper45": [1, 0, 1]}
sname = next((a for a in sys.argv[1:] if a in DIRS), "bottom")
n = np.array(DIRS[sname], float); n /= np.linalg.norm(n)
KF = f"out/dbg_keys_{sname}.pkl"
xs = part.project((n * R)[None])[0]
site = Site(xs, part.normal(xs[None])[0])
fld = build_field(part, arr, target=xs)
jet = Jet()
pl = Planner(fld, drop)
if os.path.exists(KF) and "--replan" not in sys.argv:
    keys, summ = pickle.load(open(KF, "rb"))
else:
    keys, summ = plan_delivery(pl, jet, site, part, np.zeros(3), R)
    pickle.dump((keys, summ), open(KF, "wb"))
t, x, v, a = jet.nominal(drop)
print("nominal: v_acq %.3f m/s, rise time %.1f ms, apex z %.2f mm" % (v[0, 2], t[-1] * 1e3, x[-1, 2] * 1e3))
plant = Plant(fld)
r = simulate(plant, drop, x[0], v[0], keys, part=part, record_every=2)
tr = r["traj"]
dev = tr[:, 1:4] - tr[:, 4:7]
tb = np.cumsum([summ["t_acq"], summ["t_transit"], 0.02, summ["t_press"]])
print("phase ends [ms]:", np.round(tb * 1e3, 1), " status", r["status"], "t_end %.1f ms" % (r["t"] * 1e3))
fr_n = site.n
for tq in np.concatenate([np.arange(0, tb[1], 0.04), np.arange(tb[1], r["t"], 0.004)]):
    i = np.searchsorted(tr[:, 0], tq)
    if i < len(tr):
        info = keys.info[min(np.searchsorted(keys.t, tq), len(keys.t) - 1)]
        print(f"  t={tq*1e3:5.0f} ms  dev um = {str(np.round(dev[i]*1e6)):>20}  |dev| {np.linalg.norm(dev[i])*1e6:5.0f}  "
              f"gap mm = {(part.dist(tr[i,1:4])[0]-drop.a)*1e3:5.2f}")
c = part.project(r["x"][None])[0]
print("landing error um:", np.round((c - xs) * 1e6), " | Ba max %.2f, dynamic %.2f, aspect %.3f" % (r["ba_max"], r["ba_dyn"], r["aspect_max"]))
dphi = np.angle(np.exp(1j * np.diff(keys.phi, axis=0)))
st = np.sqrt((dphi ** 2).mean(1))
big = np.where(st > 0.6)[0]
print("keyframes with rms phase jump > 0.6 rad:", [(int(i), round(float(keys.t[i]) * 1e3, 1), round(float(st[i]), 2)) for i in big])
