"""Surface quality of a finished print, split by which way the surface faces.
Up-facing skins of overhangs are the hard case (single-droplet-thick first layer).
Usage: s11_skin.py out/print[_tag] ..."""
import sys, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from scipy.ndimage import map_coordinates
from replicator.shapes import load_benchy
from replicator.printsim import true_sdf

tgt, _, _, _ = load_benchy(f"{ROOT}/assets/benchy_download/3DBenchy.stl", scale=0.4, res=0.06e-3, pole_tip=(0, 0, 10e-3))
gx, gy, gz = np.gradient(tgt.sdf.astype(np.float32), tgt.res)
surf = np.abs(tgt.sdf) < 0.75 * tgt.res
ii = np.argwhere(surf)
gn = np.maximum(np.sqrt(gx[surf] ** 2 + gy[surf] ** 2 + gz[surf] ** 2), 1e-9)
nz = gz[surf] / gn
pts = tgt.lo + ii * tgt.res - tgt.sdf[surf][:, None] * np.stack([gx[surf], gy[surf], gz[surf]], 1) / gn[:, None]     # onto the surface itself
keep = pts[:, 2] < 10e-3 - 0.1e-3
pts, nz = pts[keep], nz[keep]
for d in sys.argv[1:]:
    f = np.load(f"{ROOT}/{d}/final.npz")
    sdf, lo, res = f["sdf"].astype(np.float32), f["lo"], float(f["res"])
    dev = map_coordinates(true_sdf(sdf, res), ((pts - lo) / res).T, order=1, mode="nearest")
    out = {}
    print(f"{d}:")
    for nm, m in (("up-facing", nz > 0.5), ("side", np.abs(nz) <= 0.5), ("down-facing", nz < -0.5)):
        e = dev[m]
        out[nm] = dict(share=float(m.mean()), mean_abs_um=float(np.abs(e).mean() * 1e6), p95_um=float(np.percentile(np.abs(e), 95) * 1e6),
                       missing_100um=float((e > 0.1e-3).mean()), missing_200um=float((e > 0.2e-3).mean()))
        print(f"  {nm:12s} ({100*m.mean():4.1f} % of surface): mean |dev| {out[nm]['mean_abs_um']:4.0f} um, p95 {out[nm]['p95_um']:4.0f} um, "
              f"material missing by >100 um on {100*out[nm]['missing_100um']:4.1f} %, by >200 um on {100*out[nm]['missing_200um']:4.1f} %")
    json.dump(out, open(f"{ROOT}/{d}/skin.json", "w"), indent=1)
