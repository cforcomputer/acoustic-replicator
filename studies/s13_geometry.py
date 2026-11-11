"""Study 13: which array geometry?  Same physics, same metrics, for each candidate:
 - output efficiency (in-phase focus pressure vs an ideal sphere of the same count),
 - free-space trap stiffness under the droplet-integrity limit, across the build zone,
 - landing funnel quality on a 20 mm part at four surface orientations.
Usage: s13_geometry.py CONFIG [--vpp 18]"""
import sys, time, json, os
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import configs
from replicator.acoustics import Field
from replicator.gorkov import Droplet
from replicator.holo import TrapSolver
from replicator.scene import new_part, build_field, frame_from_normal

cfg = sys.argv[1]
VPP = float(sys.argv[sys.argv.index("--vpp") + 1]) if "--vpp" in sys.argv else 18.0
drop = Droplet(0.3e-3)
arr = configs.REGISTRY[cfg](VPP)
fld = Field(arr)
out = dict(cfg=cfg, name=arr.name, n=arr.n, vpp=VPP)
foc = np.exp(1j * arr.focus_phases(np.zeros(3)))
p0 = abs(fld.eval(np.zeros((1, 3)), foc)[0][0])
dist = np.linalg.norm(arr.pos, axis=1)
out["focus_kPa"] = p0 / 1e3
out["efficiency"] = p0 / (arr.p_ref * np.sum(1 / dist))
sv = TrapSolver(fld, drop)
ks, amps = [], []
for x in ([0, 0, 0], [12e-3, 0, 0], [0, 0, -14e-3], [9e-3, 9e-3, 8e-3], [0, 0, -30e-3], [16e-3, -10e-3, -6e-3]):
    phi, amp, info = sv.solve_best(np.array(x, float), n_random=1)
    ks.append(float(info["kappa"].min())); amps.append(amp)
out["kappa_free_min"] = min(ks); out["kappa_free_med"] = float(np.median(ks)); out["amp_free"] = float(np.mean(amps))
R = 10e-3
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
gaps = np.array([0.03, 0.3, 0.6, 0.9, 1.2]) * 1e-3
sites = {}
for sname, nv in (("bottom", [0, 0, -1]), ("side", [1, 0, 0]), ("upper45", [1, 0, 1]), ("top", [0.35, 0, 1])):
    n = np.array(nv, float); n /= np.linalg.norm(n)
    xs = part.project((n * R)[None])[0]
    fr = frame_from_normal(part.normal(xs[None])[0])
    f2 = build_field(part, arr, target=xs)
    s2 = TrapSolver(f2, drop)
    phi, s, info = s2.solve_funnel(xs, fr, gaps, n_random=2)
    p3, a3, i3 = s2.solve_best(xs + fr[2] * (drop.a + 2.0e-3), n_random=1)
    sites[sname] = dict(funnel_kt=float(info["kappa_t"].min()), funnel_fn=[float(info["f_n"].min()), float(info["f_n"].max())],
                        funnel_Ba=float(info["Ba"]), trap2mm_k=float(i3["kappa"].min()))
out["sites"] = sites
out["funnel_kt_worst"] = min(v["funnel_kt"] for v in sites.values())
out["trap2mm_worst"] = min(v["trap2mm_k"] for v in sites.values())
os.makedirs(f"{ROOT}/out/geometry", exist_ok=True)
json.dump(out, open(f"{ROOT}/out/geometry/{cfg}_v{int(VPP)}.json", "w"), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "sites"}))
