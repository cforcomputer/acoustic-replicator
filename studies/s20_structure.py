"""Study 20: does the half-built part need structural support?

For snapshots of the simulated print, hanging from the pin with nothing else holding it:
  A. own weight: sag of every point and the highest stress;
  B. how far each of the next landing sites moves per unit force (compliance), most flexible first;
  C. (with --acoustic) the radiation pressure the sound field puts on the part while a droplet is
     pressed on, from the boundary-element surface pressure of a planned delivery, applied to the
     same model: how far the landing site is pushed away.
Displacements scale as 1/E; everything is solved at E = 1 GPa and reported for a range of moduli.

Usage: s20_structure.py [print dir = out/print] [--snaps 0.030,0.100,...] [--sites 10] [--st 3]
       s20_structure.py --acoustic 0.250 autoside [--d 0.3]      (one delivery; takes minutes)
"""
import sys, json, os, time, glob
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import structure as S

args = sys.argv[1:]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
ST = opt("--st", 3)
NSITE = opt("--sites", 10)
POLE_TIP = 10e-3
E_REF = 1e9
RHO = 1150.0
fixed = lambda p: p[:, 2] > POLE_TIP + 0.05e-3


def model_of(snap_path, st=ST, verbose=True):
    d = np.load(snap_path)
    occ, lo, h = S.occupancy_from_sdf(d["sdf"].astype(np.float32), d["lo"], float(d["res"]), st=st, zmax=POLE_TIP + 0.8e-3)
    t0 = time.time()
    m = S.Model(occ, lo, h, fixed, E=E_REF, rho=RHO, verbose=verbose)
    return m, d, time.time() - t0


if "--acoustic" not in args:
    pdir = args[0] if args and not args[0].startswith("--") else "out/print"
    pdir = pdir if pdir.startswith("/") else f"{ROOT}/{pdir}"
    snaps = opt("--snaps", "")
    files = sorted(glob.glob(f"{pdir}/snap_*.npz")) if not snaps else [f"{pdir}/snap_{s}.npz" for s in snaps.split(",")]
    out = dict(E_ref_GPa=E_REF / 1e9, voxel_mm=None, rho=RHO, states=[])
    for fn in files:
        frac = os.path.basename(fn)[5:-4]
        m, d, tb = model_of(fn)
        out["voxel_mm"] = m.h * 1e3
        t0 = time.time()
        ug = m.solve(m.gravity())
        ts = time.time() - t0
        mag = np.linalg.norm(ug, axis=1)
        vm = m.von_mises(ug)
        mass = RHO * m.h ** 3 * m.occ.sum()
        imax = int(np.argmax(mag))
        # stress where the part meets the pin: elements just under the pin face
        zc = m.lo[2] + (m.ijk[:, 2] + 0.5) * m.h
        top = (zc > POLE_TIP - 0.6e-3) & (zc < POLE_TIP)
        st_rec = dict(frac=float(frac), droplets=int(d["n_dep"]) if "n_dep" in d.files else None, elements=len(m.ijk), mass_g=mass * 1e3,
                      sag_max_um=float(mag.max() * 1e6), sag_max_at_mm=[float(x) for x in m.xyz[imax] * 1e3],
                      vm_max_kPa=float(vm.max() / 1e3), vm_pin_joint_kPa=float(vm[top].max() / 1e3) if top.any() else None,
                      f1_hz_est=float(np.sqrt(S.G_ACC / max(mag.max(), 1e-12)) / (2 * np.pi)),
                      solve_s=ts, iters=m.last_iters, resid=m.last_residual, dropped_voxels=m.dropped)
        cand = d["cand"] if "cand" in d.files else np.zeros((0, 6))
        sites = []
        if len(cand):
            sag_c = np.linalg.norm(m.at(ug, cand[:, :3]), axis=1)
            order = np.argsort(-sag_c)[:NSITE]
            for k in order:
                x, n = cand[k, :3], cand[k, 3:6]
                u = m.solve(m.patch(x, -1e-6 * n, 0.35e-3))                 # 1 uN pushing into the surface
                us = m.at(u, x)[0]
                sites.append(dict(k=int(k), x_mm=[float(v) for v in x * 1e3], n=[float(v) for v in n], sag_um=float(sag_c[k] * 1e6),
                                  comp_um_per_uN=float(np.linalg.norm(us) * 1e6), comp_normal_um_per_uN=float(-(us @ n) * 1e6)))
            st_rec["cand_sag_um"] = dict(median=float(np.median(sag_c) * 1e6), p90=float(np.percentile(sag_c, 90) * 1e6), max=float(sag_c.max() * 1e6))
        st_rec["sites"] = sites
        out["states"].append(st_rec)
        cmax = max([s["comp_um_per_uN"] for s in sites], default=float("nan"))
        print(f"{os.path.basename(pdir)} {frac}: {len(m.ijk)} elements, {mass*1e3:.3f} g | own weight at E = 1 GPa: sag max {mag.max()*1e6:.3f} um, "
              f"stress max {vm.max()/1e3:.2f} kPa (pin joint {st_rec['vm_pin_joint_kPa'] or 0:.2f}) | most flexible landing site {cmax:.4f} um/uN | "
              f"build {tb:.0f}s solve {ts:.0f}s ({m.last_iters} it)", flush=True)
    tag = os.path.basename(pdir)
    json.dump(out, open(f"{ROOT}/out/structure_{tag}.json", "w"), indent=1)
    sys.exit(0)

# ---------------------------------------------------------------------------------------------
# C. radiation pressure of a planned delivery on the part
# ---------------------------------------------------------------------------------------------
from replicator import constants as C, configs
from replicator.acoustics import Scatterer, Field
from replicator.geometry import VoxelPart, refine_near
from replicator.gorkov import Droplet
from replicator.scene import POLE_RADIUS, DEFAULT_LEVELS
from replicator.dynamics import Planner
from replicator.delivery import Site, Jet, JET_MK2, plan_delivery, plan_direct

i0 = args.index("--acoustic")
frac, site_name = args[i0 + 1], args[i0 + 2]
DROP_D = opt("--d", 0.3) * 1e-3
PRINT = opt("--print", "out/print")
drop = Droplet(DROP_D)
arr = configs.REGISTRY["rco5"](18.0)
snap = np.load(f"{ROOT}/{PRINT}/snap_{frac}.npz")
sdf, lo, res = snap["sdf"].astype(np.float32), snap["lo"], float(snap["res"])
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
if site_name.startswith("auto"):
    nzc = cand[:, 5]
    k = int({"autodown": np.argmin(nzc), "autoup": np.argmax(nzc), "autoside": np.argmin(np.abs(nzc))}[site_name])
else:
    k = int(site_name)
xs = part.project(cand[k, :3][None])[0]
n_site = part.normal(xs[None])[0]
site = Site(xs, n_site)
t0 = time.time()
mesh = refine_near(part.to_mesh(h=1.0e-3, zmax=40e-3, mc=0.3e-3), xs, DEFAULT_LEVELS, project=part.project)
fld = Field(arr, Scatterer(mesh, arr))
print(f"benchy {frac} site {k} at {np.round(xs*1e3,2)} mm, normal {np.round(n_site,2)}; {mesh.n} panels, BEM {time.time()-t0:.0f}s", flush=True)
pl = Planner(fld, drop, ds=0.15e-3, a_max=0.5 * C.G_ACC)
jet = Jet(**JET_MK2)
GAP0 = 1.2e-3
x_pre = xs + n_site * (drop.a + GAP0)
t0 = time.time()
if n_site[2] < -0.6 and np.hypot(x_pre[0], x_pre[1]) < 16e-3:
    keys, summ = plan_direct(pl, jet, site, part, gap0=GAP0)
else:
    keys, summ = plan_delivery(pl, jet, site, part, center, radius, gap0=GAP0); summ["mode"] = "carry"
print(f"  planned {summ['mode']} in {time.time()-t0:.0f}s, {len(keys.t)} keyframes, {keys.t[-1]*1e3:.0f} ms", flush=True)

m, d, _ = model_of(f"{ROOT}/{PRINT}/snap_{frac}.npz")
below = mesh.cen[:, 2] < POLE_TIP                      # panels on the part (the rest are on the steel pin)
near = np.linalg.norm(mesh.cen - xs, axis=1)
rows = []
best = None
Pinc, _ = arr.rows(mesh.cen, grad=False)                # incident field at the panels, per transducer
clipped = 0
for i in range(len(keys.t)):
    w = keys.amp[i] * np.exp(1j * keys.phi[i])
    ps = fld.scat.M @ w
    cap = 3.0 * float(np.abs(Pinc @ w).max())
    P, F, nc = S.radiation_pressure(mesh, ps, arr.k, p_cap=cap)
    clipped = max(clipped, nc)
    Fp = F[below]
    rec = dict(t_ms=float(keys.t[i] * 1e3), gap_mm=float(np.linalg.norm(keys.xd[i] - xs) * 1e3), amp=float(keys.amp[i]),
               p_surf_max_kPa=float(min(np.abs(ps[below]).max(), cap) / 1e3), p_inc_max_kPa=cap / 3e3, clipped_panels=nc, P_rad_max_Pa=float(P[below].max()), P_rad_min_Pa=float(P[below].min()),
               F_net_uN=[float(v) for v in Fp.sum(0) * 1e6], F_near2mm_uN=[float(v) for v in F[below & (near < 2e-3)].sum(0) * 1e6],
               F_abs_sum_uN=float(np.linalg.norm(Fp, axis=1).sum() * 1e6))
    rows.append(rec)
    if best is None or np.linalg.norm(rec["F_net_uN"]) > np.linalg.norm(best[0]["F_net_uN"]):
        best = (rec, F.copy(), i)
# apply the load of every 3rd keyframe (and the worst) to the structure; record the site's displacement
disp = []
idx = sorted(set(list(range(0, len(keys.t), max(1, len(keys.t) // 14))) + [best[2], len(keys.t) - 1]))
for i in idx:
    w = keys.amp[i] * np.exp(1j * keys.phi[i])
    P, F, _ = S.radiation_pressure(mesh, fld.scat.M @ w, arr.k, p_cap=3.0 * float(np.abs(Pinc @ w).max()))
    u = m.solve(m.panel_loads(mesh.cen[below], F[below]))
    us = m.at(u, xs)[0]
    disp.append(dict(t_ms=float(keys.t[i] * 1e3), gap_mm=float(np.linalg.norm(keys.xd[i] - xs) * 1e3), site_disp_um=[float(v) for v in us * 1e6],
                     site_disp_abs_um=float(np.linalg.norm(us) * 1e6), part_disp_max_um=float(np.linalg.norm(u, axis=1).max() * 1e6),
                     vm_max_kPa=float(m.von_mises(u).max() / 1e3)))
    print(f"  t {keys.t[i]*1e3:6.1f} ms, droplet {disp[-1]['gap_mm']:5.2f} mm from the site: net force on part {np.linalg.norm(rows[i]['F_net_uN']):7.2f} uN, "
          f"peak radiation pressure {rows[i]['P_rad_max_Pa']:6.1f} Pa, site moves {disp[-1]['site_disp_abs_um']:.4f} um, part max {disp[-1]['part_disp_max_um']:.4f} um (E = 1 GPa)", flush=True)
ug = m.solve(m.gravity())
res_out = dict(panels=int(mesh.n), clipped_panels_max=int(clipped), frac=float(frac), site=k, x_mm=[float(v) for v in xs * 1e3], n=[float(v) for v in n_site], mode=summ["mode"], droplet_mm=DROP_D * 1e3,
               droplet_weight_uN=float(drop.mass * C.G_ACC * 1e6), E_ref_GPa=1.0, keyframes=rows, displacement=disp,
               gravity_sag_site_um=float(np.linalg.norm(m.at(ug, xs)[0]) * 1e6), part_weight_uN=float(RHO * m.h ** 3 * m.occ.sum() * S.G_ACC * 1e6))
json.dump(res_out, open(f"{ROOT}/out/structure_acoustic_{frac}_{site_name}_d{int(DROP_D*1e6)}.json", "w"), indent=1)
F_all = np.array([np.linalg.norm(r["F_net_uN"]) for r in rows])
print(f"  net radiation force on the part: max {F_all.max():.1f} uN, at landing {F_all[-1]:.1f} uN; droplet weight {res_out['droplet_weight_uN']:.3f} uN; part weight {res_out['part_weight_uN']:.0f} uN")
print(f"  landing site displacement at E = 1 GPa: max {max(x['site_disp_abs_um'] for x in disp):.4f} um, at landing {disp[-1]['site_disp_abs_um']:.4f} um; own-weight sag there {res_out['gravity_sag_site_um']:.4f} um")
