import sys, pickle
import numpy as np
sys.path.insert(0, ".")
from replicator import constants as C, configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.dynamics import Plant, simulate
from replicator.delivery import Jet

R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
n = np.array([1.0, 0, 0])
xs = part.project((n * R)[None])[0]
fld = build_field(part, arr, target=xs)
keys, summ = pickle.load(open("out/dbg_keys_side.pkl", "rb"))
jet = Jet()
t, x, v, a = jet.nominal(drop)
cases = {
    "phase 3deg only": dict(phase_err=np.radians(3) * np.random.default_rng(1).normal(size=arr.n)),
    "amp 5% only": dict(amp_err=0.05 * np.random.default_rng(2).normal(size=arr.n)),
    "quant 64 only": dict(phase_levels=64),
    "wind 0.2 +z": dict(wind=[0, 0, 0.2]),
    "wind 0.2 -x": dict(wind=[-0.2, 0, 0]),
    "phase 10deg": dict(phase_err=np.radians(10) * np.random.default_rng(3).normal(size=arr.n)),
    "phase 3deg b": dict(phase_err=np.radians(3) * np.random.default_rng(4).normal(size=arr.n)),
}
for nm, kw in cases.items():
    plant = Plant(fld, **kw)
    r = simulate(plant, drop, x[0], v[0], keys, part=part, record_every=2, t_extra=0.06)
    tr = r["traj"]
    msg = f"{nm:18s}: {r['status']:8s} t={r['t']*1e3:.0f} ms, maxBa {r['ba_max']:.2f} dyn {r['ba_dyn']:.2f}"
    if r["status"] == "landed":
        c = part.project(r["x"][None])[0]
        msg += f", err um {np.round((c - xs)*1e6)}"
    else:
        tail = tr[-1]
        msg += f", final gap {(part.dist(tail[1:4])[0]-drop.a)*1e3:.2f} mm, final dev um {np.round((tail[1:4]-tail[4:7])*1e6)}"
        i0 = np.searchsorted(tr[:, 0], 0.270)
        for i in range(i0, len(tr), 12):
            print(f"      t={tr[i,0]*1e3:5.1f} gap={(part.dist(tr[i,1:4])[0]-drop.a)*1e3:5.2f} mm  dev um={np.round((tr[i,1:4]-tr[i,4:7])*1e6)}")
    print(msg, flush=True)
