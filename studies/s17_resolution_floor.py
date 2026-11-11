"""How well could ANY process that builds from round blobs of a given size reproduce the target?

Round every convex and concave feature of the target with a ball of radius r (morphological
opening then closing) and measure how far that idealised shape is from the true target, on
edges and on faces.  This is the part of the edge error that comes from the size of the
building block and cannot be removed by better placement.
Usage: s17_resolution_floor.py [--zcut_mm 3.0]
"""
import sys, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/studies")
from scipy.ndimage import distance_transform_edt, map_coordinates
from replicator.shapes import load_benchy
from s15_edges import classify, report

args = sys.argv[1:]
zcut = float(args[args.index("--zcut_mm") + 1]) * 1e-3 if "--zcut_mm" in args else None
tgt, _, _, _ = load_benchy(f"{ROOT}/assets/benchy_download/3DBenchy.stl", scale=0.4, res=0.06e-3, pole_tip=(0, 0, 10e-3))
pts, nrm, edge = classify(tgt, z_max=10e-3 - 0.1e-3, z_min=None if zcut is None else zcut + 0.15e-3)
occ = tgt.sdf < 0
res = tgt.res
out = {}
for r in (0.05e-3, 0.10e-3, 0.15e-3, 0.20e-3, 0.27e-3):
    opened = distance_transform_edt(~(distance_transform_edt(occ) * res > r)) * res <= r            # rounds convex edges, drops features thinner than 2r
    closed = ~(distance_transform_edt(~(distance_transform_edt(~opened) * res > r)) * res <= r)     # rounds concave corners, fills gaps narrower than 2r
    sdf = np.where(closed, -(distance_transform_edt(closed) - 0.5), distance_transform_edt(~closed) - 0.5).astype(np.float32) * res
    dev = map_coordinates(sdf, ((pts - tgt.lo) / res).T, order=1, mode="nearest")
    rep = report(dev, edge)
    out[f"{r*1e6:.0f}"] = rep
    print(f"rounding radius {r*1e6:3.0f} um: edges mean |dev| {rep['edges']['mean_abs_um']:4.0f} um (p95 {rep['edges']['p95_um']:4.0f}), faces {rep['faces']['mean_abs_um']:4.0f} um, "
          f"edge points missing by > 100 um: {100*rep['edges']['missing_100um']:4.1f} %", flush=True)
json.dump(out, open(f"{ROOT}/out/resolution_floor.json", "w"), indent=1)
