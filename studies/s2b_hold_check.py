"""Sanity: a droplet released at the trap equilibrium of a static hologram must stay put;
released 100 um off it must oscillate at the predicted trap frequency."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from replicator import configs
from replicator.acoustics import Field
from replicator.gorkov import Droplet
from replicator.dynamics import Planner, Plant, Keys, simulate

drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
fld = Field(arr)
pl = Planner(fld, drop)
x = np.array([5e-3, -3e-3, 2e-3])
k = pl.hold(x, 0.25, None, 0.5)
info = k.info[0]
print("kappa", np.round(info["kappa"], 2), "freq", np.round(info["freq_hz"], 1), "offset um", np.round(info["offset"] * 1e6, 1))
plant = Plant(fld)
for off in (np.zeros(3), np.array([100e-6, 0, 0])):
    r = simulate(plant, drop, x + info["offset"] + off, np.zeros(3), k, t_extra=0, record_every=1)
    tr = r["traj"]
    d = tr[:, 1:4] - (x + info["offset"])
    amp = np.abs(d).max(0) * 1e6
    # dominant frequency along x
    sig = d[:, 0] - d[:, 0].mean()
    f = np.fft.rfftfreq(len(sig), tr[1, 0] - tr[0, 0])
    fpk = f[np.abs(np.fft.rfft(sig)).argmax()]
    print(f"release offset {off[0]*1e6:.0f} um: max |dev| xyz = {np.round(amp,1)} um, dominant freq {fpk:.1f} Hz, "
          f"end dev {np.round(d[-1]*1e6,1)} um")
