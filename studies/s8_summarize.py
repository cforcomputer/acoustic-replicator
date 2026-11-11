"""Collect every Monte-Carlo result into one table (and the viewer status file)."""
import sys, json, glob, os
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import status
rows = []
for fn in sorted(glob.glob(f"{ROOT}/out/mc/*.json")):
    d = json.load(open(fn))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["plant"], []).append(r)
    allr = d["rows"]
    ok = [r for r in allr if r["status"] == "landed"]
    e = np.array([r["err_true"] for r in ok]) * 1e6 if ok else np.array([np.nan])
    per = {}
    for p, rr in by.items():
        okp = [r["err_true"] * 1e6 for r in rr if r["status"] == "landed"]
        per[p] = (len(okp), len(rr), float(np.median(okp)) if okp else float("nan"))
    rows.append(dict(key=os.path.basename(fn)[:-5], cfg=d["cfg"], scene=d["scene"], site=d["site"], d=d["d"] * 1e3,
                     landed=len(ok), n=len(allr), med=float(np.median(e)), p90=float(np.percentile(e, 90)) if ok else np.nan,
                     worst=float(np.nanmax(e)), ba=max(r.get("ba_dyn", r["ba"]) for r in allr), per=per,
                     t=d["summary"]["t_total"], kmin=d["summary"]["kappa_min"], model_R=d.get("model_R", 0),
                     mode=d["summary"].get("mode", "carry")))
print(f"{'case':46s} {'landed':>8s} {'median':>7s} {'p90':>6s} {'worst':>6s} {'Ba_dyn':>6s} {'flight':>7s}  per-plant landed/median um")
for r in rows:
    per = "  ".join(f"{p}:{a}/{b}/{m:.0f}" for p, (a, b, m) in r["per"].items())
    print(f"{r['key']:46s} {r['landed']:3d}/{r['n']:<3d} {r['med']:7.0f} {r['p90']:6.0f} {r['worst']:6.0f} {r['ba']:6.2f} {r['t']*1e3:5.0f}ms {r['mode'][:6]:6s} {per}")
json.dump(rows, open(f"{ROOT}/out/mc_summary.json", "w"), indent=1)
if "--status" in sys.argv:
    for r in rows:
        state = "pass" if r["landed"] == r["n"] and r["med"] < 300 else "warn" if r["landed"] >= 0.8 * r["n"] else "fail"
        status.update(test=(r["key"], f"{r['cfg']} · {r['scene']} · {r['site']}", state,
                            f"landed {r['landed']}/{r['n']} · median error {r['med']:.0f} µm · worst {r['worst']:.0f} µm\n"
                            f"flight {r['t']*1e3:.0f} ms · peak dynamic Bond {r['ba']:.2f} (breaks up ≈2.3)"))
