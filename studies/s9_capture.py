"""Study 9: capture margin -- how wrong can the jet be before a droplet is lost?

One planned delivery (side of the 20 mm part), ideal acoustics, and a jet whose
exit speed / direction / timing is deliberately wrong by increasing amounts,
with NO light-gate correction.  Reports where droplets stop landing.
"""
import sys, time, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import configs, status
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.dynamics import Planner, Plant, simulate
from replicator.delivery import Site, Jet, plan_delivery

R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
n = np.array([1.0, 0, 0])
xs = part.project((n * R)[None])[0]
fld = build_field(part, arr, target=xs)
jet = Jet()
pl = Planner(fld, drop, ds=0.15e-3)
keys, summ = plan_delivery(pl, jet, Site(xs, part.normal(xs[None])[0]), part, np.zeros(3), R)
plant = Plant(fld)
rows = []


def shot(label, **kw):
    x0, v0 = jet.state(drop, **kw)
    r = simulate(plant, drop, x0, v0, keys, part=part)
    err = float(np.linalg.norm(part.project(r["x"][None])[0] - xs)) * 1e6 if r["status"] == "landed" else None
    rows.append(dict(label=label, status=r["status"], err_um=err, ba_dyn=r["ba_dyn"], **{k: (list(v) if hasattr(v, "__len__") else v) for k, v in kw.items()}))
    print(f"  {label:34s} -> {r['status']:7s}" + (f" {err:5.0f} um off target" if err is not None else "") + f"  (peak dynamic Bond {r['ba_dyn']:.2f})", flush=True)


print("exit-speed error (typical jet: 1 % rms)")
for e in (0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12):
    shot(f"speed {e:+d} %", speed_err=e / 100)
print("pointing error (typical jet: 0.3 deg rms)")
for a in (0.5, 1, 2, 3, 4, 6):
    shot(f"tilt {a} deg", angle=(np.radians(a), 0.0))
print("timing error (typical: < 0.05 ms)")
for t in (0.2, 0.5, 1, 2, 4):
    shot(f"late by {t} ms", time_err=t * 1e-3)
    shot(f"early by {t} ms", time_err=-t * 1e-3)
json.dump(rows, open(f"{ROOT}/out/capture_margin.json", "w"), indent=1)


def limit(key, vals):
    ok = [abs(v) for v, r in vals if r["status"] == "landed"]
    bad = [abs(v) for v, r in vals if r["status"] != "landed"]
    return (max(ok) if ok else 0), (min(bad) if bad else None)


sp = [(r["speed_err"] * 100, r) for r in rows if "speed_err" in r]
an = [(np.degrees(r["angle"][0]), r) for r in rows if "angle" in r]
ti = [(r["time_err"] * 1e3, r) for r in rows if "time_err" in r]
ls, la, lt = limit("s", sp), limit("a", an), limit("t", ti)
msg = (f"lands up to ±{ls[0]:.0f} % speed error" + (f" (lost at {ls[1]:.0f} %)" if ls[1] else " (never lost in the range tested)")
       + f" · up to {la[0]:.1f}° pointing error" + (f" (lost at {la[1]:.0f}°)" if la[1] else " (never lost)")
       + f"\\nup to {lt[0]:.1f} ms timing error" + (f" (lost at {lt[1]:.1f} ms)" if lt[1] else " (never lost)")
       + " · typical jet: 1 %, 0.3°, <0.05 ms")
print(msg.replace("\\n", "\n"))
status.update(test=("capture_margin", "Capture margin: how wrong can the jet be before a droplet is lost? (no gate correction)",
                    "pass" if ls[0] >= 3 and la[0] >= 1 else "warn", msg.replace("\\n", "\n")))
