"""Study 3: Monte-Carlo delivery accuracy with an imperfect machine.

Plan the hologram sequence on the controller's model, then fly the droplet in
several perturbed "true" machines.  The part is either a test sphere on the
pole or a snapshot of the growing Benchy print.

  s3_mc.py CONFIG SCENE SITE [options]
     SCENE: tip | sph10 | sph20 | sph30 | benchy:<snapshot fraction>
     SITE : bottom | lower45 | side | farside | upper45 | top | cand<k> (benchy)
  options: --d mm  --n draws  --vpp V  --tols cal,uncal  --plants nominal,dT+,dT-,R0.3,R0.6,B200 (boards 0.2 mm out of place)
           --model_R r   controller's model includes wall reflections r
           --ds mm       hologram keyframe spacing   --mu Pa.s  resin viscosity
           --tilt deg    approach tilted from the normal (oblique funnel)   --tag x
"""
import sys, time, json, os, re
import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import constants as C
from replicator import configs, status
from replicator.acoustics import Mesh, Scatterer, Field
from replicator.geometry import VoxelPart
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field, frame_from_normal, POLE_RADIUS
from replicator.dynamics import Planner, Plant, simulate
from replicator.delivery import Site, Jet, JET_MK2, plan_delivery, plan_direct

args = sys.argv[1:]
cfg, scene, site_name = args[0], args[1], args[2]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
DROP_D = opt("--d", 0.3) * 1e-3
NDRAW = opt("--n", 2)
VPP = opt("--vpp", 20.0)
TAG = opt("--tag", "")
DS = opt("--ds", 0.15) * 1e-3
MODEL_R = opt("--model_R", 0.0)
MU = opt("--mu", C.MU_RESIN)
TILT = np.radians(opt("--tilt", 0.0))
DIRECT = opt("--direct", "auto")              # auto: direct ascent for sites the jet can see from below
AMAX = opt("--amax", 0.5)                     # peak acceleration of the carry move [g]
GAP0 = opt("--gap0", 1.2) * 1e-3               # stand-off where the 3-D trap hands over to the funnel
BOTTOM_GAIN = opt("--bottom_gain", 1.0)        # amplitude transmission of the drip screen over the bottom board
DT = opt("--dT", 0.5)
TOLS = opt("--tols", "cal").split(",")
PLANTS = opt("--plants", "nominal,dT+,R0.3,R0.6").split(",")

SCENES = {"tip": (1.5e-3, 8e-3), "sph10": (5e-3, 10e-3), "sph20": (10e-3, 16e-3), "sph30": (15e-3, 21e-3)}
DIRS = {"bottom": [0, 0, -1], "lower45": [1, 0, -1], "side": [1, 0, 0], "farside": [-1, 0, 0],
        "upper45": [1, 0, 1], "top": [0.35, 0, 1]}
TOL = {"cal": (3.0, 0.05, 128, 0.3, True),      # after self-calibration with the on-board microphones
       "raw": (14.0, 0.15, 128, 0.3, True),     # generic transducers as delivered, no calibration
       "uncal": (10.0, 0.15, 32, 0.3, False)}
#        phase sd [deg], amp sd, phase levels, streaming wind max [m/s], droplet light gate

drop = Droplet(DROP_D, mu=MU)
def make_array():
    a = configs.REGISTRY[cfg](VPP)
    if BOTTOM_GAIN != 1.0:
        a.set_panel_gain("bottom", BOTTOM_GAIN)
    return a


arr = make_array()
key = f"{cfg}_{scene.replace(':', '')}_{site_name}_d{int(DROP_D*1e6)}{TAG}"
title = f"{cfg} · {scene} · {site_name}" + (f" · {TAG}" if TAG else "")
status.update(test=(key, title, "running", "planning holograms…"))

if scene.startswith("benchy"):
    frac = scene.split(":")[1]
    snap = np.load(f"{ROOT}/out/print/snap_{frac}.npz")
    sdf, lo, res = snap["sdf"].astype(np.float32), snap["lo"], float(snap["res"])
    # coarser copy for the acoustics, extended upward to carry the pole
    st = 3
    cs = sdf[::st, ::st, ::st]
    hi = lo + res * st * (np.array(cs.shape) - 1)
    part = VoxelPart(lo, [hi[0], hi[1], 45e-3], res=res * st, band=0.5e-3)
    nz = cs.shape[2]
    part.sdf[: cs.shape[0], : cs.shape[1], :nz] = np.minimum(part.sdf[: cs.shape[0], : cs.shape[1], :nz], cs)
    part.add_cylinder([0, 0, 10e-3], [0, 0, 47e-3], POLE_RADIUS)
    solid = np.argwhere(part.sdf[:, :, :nz] < 0) * part.res + part.lo
    center = 0.5 * (solid.min(0) + solid.max(0))
    radius = float(np.linalg.norm(solid - center, axis=1).max())
    cand = snap["cand"]
    k = int(re.sub(r"\D", "", site_name) or 0) % max(len(cand), 1)
    if site_name.startswith("auto"):                        # pick by orientation: autodown / autoside / autoup
        nzc = cand[:, 5]
        k = int({"autodown": np.argmin(nzc), "autoup": np.argmax(nzc), "autoside": np.argmin(np.abs(nzc))}[site_name])
    xs = part.project(cand[k, :3][None])[0]
    n_site = part.normal(xs[None])[0]
    mesh_kw = dict(h=1.0e-3)
    R = 0.0
else:
    R, ext = SCENES[scene]
    n0 = np.array(DIRS[site_name], float); n0 /= np.linalg.norm(n0)
    part = new_part(extent=ext, pole_tip_z=max(R - 2e-3, 0.0), res=0.2e-3 if R > 6e-3 else 0.1e-3)
    part.add_sphere([0, 0, 0], R)
    xs = part.project((n0 * R)[None])[0]
    n_site = part.normal(xs[None])[0]
    center, radius = np.zeros(3), R
    mesh_kw = {}

n_app = n_site
if TILT > 0:                                               # oblique final approach
    fr = frame_from_normal(n_site)
    n_app = np.cos(TILT) * n_site + np.sin(TILT) * fr[0]
site = Site(xs, n_app, n_surf=n_site if TILT > 0 else None)

arr_model = arr.with_reflections(MODEL_R) if MODEL_R > 0 else arr
t0 = time.time()
if scene.startswith("benchy"):
    from replicator.geometry import refine_near
    from replicator.scene import DEFAULT_LEVELS
    mesh = refine_near(part.to_mesh(h=1.0e-3, zmax=40e-3, mc=0.3e-3), xs, DEFAULT_LEVELS, project=part.project)
    fld = Field(arr_model, Scatterer(mesh, arr_model))
else:
    fld = build_field(part, arr_model, target=xs)
print(f"[{title}] {arr.n} transducers, {fld.scat.mesh.n} panels, BEM {time.time()-t0:.1f}s; site normal "
      f"{np.round(n_site,2)}", flush=True)

central = cfg.startswith("rco") or cfg.startswith("sphere")
jet = Jet(**JET_MK2) if central else Jet()
pl = Planner(fld, drop, ds=DS, a_max=AMAX * C.G_ACC)
t0 = time.time()
x_pre = xs + n_site * (drop.a + GAP0)
use_direct = central and DIRECT != "off" and (DIRECT == "on" or (n_app[2] < -0.6 and np.hypot(x_pre[0], x_pre[1]) < 16e-3))
if use_direct:
    keys, summ = plan_direct(pl, jet, site, part, gap0=GAP0)
else:
    keys, summ = plan_delivery(pl, jet, site, part, center, radius, gap0=GAP0)
    summ["mode"] = "carry"
print(f"  plan {time.time()-t0:.0f}s: " + ", ".join(f"{k}={v:.3g}" if not isinstance(v, str) else f"{k}={v}" for k, v in summ.items()), flush=True)


class Shifted:
    """The true part, displaced by delta relative to the controller's model."""
    def __init__(self, part, delta): self.part, self.delta = part, np.asarray(delta, float)
    def dist(self, pts): return self.part.dist(np.atleast_2d(pts) - self.delta)
    def project(self, pts): return self.part.project(np.atleast_2d(pts) - self.delta) + self.delta


PL = {"nominal": (0.0, (0, 0, 0), None), "dT+": (DT, (40e-6, -30e-6, 30e-6), None),
      "dT-": (-DT, (-35e-6, 40e-6, -40e-6), None)}
BOARD = {}                                                  # "B200": every board sits 0.2 mm (rms per axis) from where the model has it
for nm in PLANTS:
    if nm.startswith("R"):
        PL[nm] = (0.0, (0, 0, 0), float(nm[1:]))
    if nm.startswith("B"):
        PL[nm] = (0.0, (0, 0, 0), None); BOARD[nm] = float(nm[1:]) * 1e-6
plants = []
m0 = fld.scat.mesh
for nm in PLANTS:
    dT, delta, Rw = PL[nm]
    Rw = MODEL_R if Rw is None else Rw                      # non-R plants share the controller's wall model
    if nm in BOARD:
        arr2 = make_array()
        rb = np.random.default_rng(12345)
        for j in range(len(arr2.panels)):
            off = rb.normal(0, BOARD[nm], 3)
            arr2.pos[arr2.panel_id == j] += off; arr2.sub[arr2.panel_id == j] += off
        if Rw > 0:
            arr2 = arr2.with_reflections(Rw)
        plants.append((nm, Field(arr2, Scatterer(Mesh(m0.verts, m0.faces), arr2)), Shifted(part, delta)))
        continue
    if dT == 0 and Rw == MODEL_R:
        plants.append((nm, fld, Shifted(part, delta)))
        continue
    arr2 = make_array()
    arr2.k = C.K * (1 - dT / (2 * 293.15))
    if Rw > 0:
        arr2 = arr2.with_reflections(Rw)
    plants.append((nm, Field(arr2, Scatterer(Mesh(m0.verts + np.asarray(delta), m0.faces), arr2)), Shifted(part, delta)))

rng = np.random.default_rng(abs(hash(key)) % 2 ** 31)
rows, replay = [], []
for tol in TOLS:
    psd, asd, lev, wmax, gate = TOL[tol]
    for plabel, pf, ppart in plants:
        for d in range(NDRAW):
            wdir = rng.normal(size=3); wdir /= np.linalg.norm(wdir)
            plant = Plant(pf, phase_err=np.radians(psd) * rng.normal(size=arr.n),
                          amp_err=asd * rng.normal(size=arr.n), phase_levels=lev,
                          wind=wdir * wmax * rng.random())
            x0, v0 = jet.sample(drop, rng, gate=gate)
            t0 = time.time()
            r = simulate(plant, drop, x0, v0, keys, part=ppart)
            row = dict(tol=tol, plant=plabel, status=r["status"], t=r["t"], ba=r["ba_max"],
                       ba_dyn=r["ba_dyn"], aspect=r["aspect_max"])
            if r["status"] == "landed":
                c = ppart.project(r["x"][None])[0]
                row["err_true"] = float(np.linalg.norm(c - (xs + ppart.delta)))
                row["impact"] = float(np.linalg.norm(r["v"]))
            rows.append(row)
            tr = r["traj"]
            step = max(1, len(tr) // 140)
            replay.append(dict(label=f"{tol}, {plabel} #{d}", status=r["status"],
                               err_um=row.get("err_true", None) and row["err_true"] * 1e6,
                               pts=np.round(tr[::step, :4], 6).tolist()))
            print(f"  {tol:5s} {plabel:8s} #{d}: {r['status']:7s} t={r['t']*1e3:4.0f}ms "
                  f"err={row.get('err_true', float('nan'))*1e6:6.0f}um Ba_dyn={r['ba_dyn']:.2f} ({time.time()-t0:.0f}s)", flush=True)


def stats(rr):
    ok = [r for r in rr if r["status"] == "landed"]
    if not ok:
        return f"landed 0/{len(rr)}"
    e = np.array([r["err_true"] for r in ok]) * 1e6
    return (f"landed {len(ok)}/{len(rr)}; err um median {np.median(e):.0f} p90 {np.percentile(e, 90):.0f} "
            f"max {e.max():.0f}; Ba_dyn max {max(r['ba_dyn'] for r in rr):.2f}")


os.makedirs(f"{ROOT}/out/mc", exist_ok=True)
os.makedirs(f"{ROOT}/out/viewer", exist_ok=True)
json.dump(dict(cfg=cfg, scene=scene, site=site_name, d=DROP_D, vpp=VPP, n_transducers=arr.n, model_R=MODEL_R,
               ds=DS, mu=MU, tilt=float(TILT), site_normal=n_site.tolist(), summary=summ, rows=rows),
          open(f"{ROOT}/out/mc/{key}.json", "w"), indent=1, default=float)
if not scene.startswith("benchy"):
    stepk = max(1, len(keys.xd) // 150)
    json.dump(dict(scene=scene, site=site_name, R=R, plan=np.round(keys.xd[::stepk], 6).tolist(), runs=replay),
              open(f"{ROOT}/out/viewer/replay_{key}.json", "w"))
for tol in TOLS:
    print(f"  == {tol}: " + stats([r for r in rows if r["tol"] == tol]))
for plabel, _, _ in plants:
    print(f"  -- {plabel:8s}: " + stats([r for r in rows if r["plant"] == plabel]))
ok = [r for r in rows if r["status"] == "landed"]
e = np.array([r["err_true"] for r in ok]) * 1e6 if ok else np.array([np.nan])
state = "pass" if len(ok) == len(rows) and np.median(e) < 300 else "warn" if len(ok) >= 0.8 * len(rows) else "fail"
status.update(test=(key, title, state,
                    f"landed {len(ok)}/{len(rows)} · median error {np.median(e):.0f} µm · worst {np.nanmax(e):.0f} µm\n"
                    f"{summ.get('mode', 'carry')} delivery, flight {summ['t_total']*1e3:.0f} ms · peak dynamic Bond {max(r['ba_dyn'] for r in rows):.2f} (breaks up ≈2.3)"))
