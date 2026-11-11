"""Edge fidelity of a finished print.

Splits the target surface into "edge" points (where the surface normal turns
sharply within a quarter of a millimetre: rims, corners, window frames) and
"face" points (everything else), and reports how far the printed surface is
from the target on each.  Positive deviation = material missing, negative =
material proud of the target.

Usage: s15_edges.py out/print[_tag] ... [--zcut_mm z]
"""
import sys, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from scipy.ndimage import map_coordinates, gaussian_filter
from replicator.shapes import load_benchy
from replicator.printsim import true_sdf


def classify(tgt, radius=0.25e-3, z_max=None, z_min=None):
    """Target surface points, unit normals, and the edge flag."""
    g = np.stack(np.gradient(tgt.sdf.astype(np.float32), tgt.res))
    g /= np.maximum(np.linalg.norm(g, axis=0), 1e-9)
    surf = np.abs(tgt.sdf) < 0.75 * tgt.res
    # coherence of the normal field over a ball of `radius`, measured on the surface shell only
    w = gaussian_filter(surf.astype(np.float32), radius / tgt.res / 2)
    m = np.stack([gaussian_filter(g[d] * surf, radius / tgt.res / 2) for d in range(3)]) / np.maximum(w, 1e-6)
    coh = np.linalg.norm(m, axis=0)[surf]
    ii = np.argwhere(surf)
    nrm = g[:, surf].T
    pts = tgt.lo + ii * tgt.res - tgt.sdf[surf][:, None] * nrm        # voxel centres moved onto the surface itself
    keep = np.ones(len(pts), bool)
    if z_max is not None: keep &= pts[:, 2] < z_max
    if z_min is not None: keep &= pts[:, 2] > z_min
    return pts[keep], nrm[keep], coh[keep] < 0.80


def report(dev, edge):
    out = {}
    for nm, m in (("edges", edge), ("faces", ~edge), ("all", np.ones(len(dev), bool))):
        e = dev[m]
        out[nm] = dict(share=float(m.mean()), mean_abs_um=float(np.abs(e).mean() * 1e6), rms_um=float(np.sqrt((e ** 2).mean()) * 1e6),
                       p95_um=float(np.percentile(np.abs(e), 95) * 1e6), mean_signed_um=float(e.mean() * 1e6),
                       missing_100um=float((e > 0.1e-3).mean()), proud_100um=float((e < -0.1e-3).mean()))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    zcut = float(args[args.index("--zcut_mm") + 1]) * 1e-3 if "--zcut_mm" in args else None
    dirs = [a for a in args if a.startswith("out/")]
    tgt, _, _, _ = load_benchy(f"{ROOT}/assets/benchy_download/3DBenchy.stl", scale=0.4, res=0.06e-3, pole_tip=(0, 0, 10e-3))
    pts, nrm, edge = classify(tgt, z_max=10e-3 - 0.1e-3, z_min=None if zcut is None else zcut + 0.15e-3)
    print(f"target surface: {len(pts)} points, {100*edge.mean():.1f} % classed as edges")
    for d in dirs:
        f = np.load(f"{ROOT}/{d}/final.npz")
        sdf, lo, res = f["sdf"].astype(np.float32), f["lo"], float(f["res"])
        dev = map_coordinates(true_sdf(sdf, res), ((pts - lo) / res).T, order=1, mode="nearest")
        r = report(dev, edge)
        json.dump(r, open(f"{ROOT}/{d}/edges.json", "w"), indent=1)
        print(f"{d}:")
        for nm in ("edges", "faces", "all"):
            x = r[nm]
            print(f"  {nm:6s} ({100*x['share']:5.1f} %): mean |dev| {x['mean_abs_um']:4.0f} um, rms {x['rms_um']:4.0f}, p95 {x['p95_um']:4.0f}, "
                  f"mean signed {x['mean_signed_um']:+4.0f}; missing >100 um on {100*x['missing_100um']:4.1f} %, proud >100 um on {100*x['proud_100um']:4.1f} %")
