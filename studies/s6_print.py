"""Study 6: print a 40 %-scale 3DBenchy droplet by droplet, hanging from the pole.

Every droplet is deposited with the capillary/cure model; its landing error is
drawn from the acoustic Monte-Carlo statistics.  Snapshots of the growing part
are saved so the full acoustic physics can be re-run on the real geometry.

Usage: s6_print.py [--sigma_um 85] [--fail 0.005] [--d 0.3] [--scale 0.4] [--delay_ms 8] [--tag x]
                   [--skin fine] [--skin_d 0.2] [--shell_um 200] [--skin_sigma_um 70] [--zcut_mm z]

--skin fine   : "fine shell" mode.  The core of the part is built from the normal
                droplets; the outer shell (--shell_um deep) is built from smaller
                droplets (--skin_d) on their own finer lattice, so edges and faces
                follow the target in smaller steps.  Volume is conserved in both.
--edges_only  : with --skin fine, use the small droplets only within --edge_r_um of a sharp
                edge or corner of the target (and --shell_um of the surface).
--zcut_mm z   : print only the part of the target above z (quick tests).
"""
import sys, time, json, os, heapq
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replicator import constants as C
from replicator.gorkov import Droplet
from replicator.shapes import load_benchy
from replicator.printsim import (Lattice, MixedLattice, PrintedPart, cure_contact_angle, cap_sphere_radius,
                                 tilted_directions, true_sdf)
from replicator.scene import POLE_RADIUS, frame_from_normal
from scipy.ndimage import map_coordinates

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
args = sys.argv[1:]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
SIGMA = opt("--sigma_um", 85.0) * 1e-6          # landing error, 1-sigma per tangential axis
P_FAIL = opt("--fail", 0.005)                   # fraction of droplets lost
DROP_D = opt("--d", 0.3) * 1e-3
SCALE = opt("--scale", 0.4)
DELAY = opt("--delay_ms", 8.0) * 1e-3
TAG = opt("--tag", "")
LAYER_H = opt("--layer_um", 150.0) * 1e-6
MODEL = opt("--model", "capillary")             # landing model: capillary (wetting equilibrium) | sphere (rigid cap)
RES = opt("--res_um", 60.0) * 1e-6
SKIN = opt("--skin", "none")                    # none | fine
SKIN_D = opt("--skin_d", 0.2) * 1e-3            # droplet diameter used for the shell
SHELL = opt("--shell_um", 200.0) * 1e-6         # depth of the shell built from the small droplets
SKIN_SIGMA = opt("--skin_sigma_um", SIGMA * 1e6) * 1e-6   # landing error of the shell droplets (careful, slower landings)
ERR_POOL = opt("--err_pool", "")                   # JSON of landing errors measured in the acoustic tests (out/mc_error_pools.json):
                                                   # errors and losses are then resampled from it by droplet size and site orientation
EDGES_ONLY = "--edges_only" in args                # with --skin fine: small droplets only near sharp edges of the target
EDGE_R = opt("--edge_r_um", 300.0) * 1e-6
ZCUT = opt("--zcut_mm", -1e3) * 1e-3
POLE_TIP = np.array([0.0, 0.0, 10e-3])
rng = np.random.default_rng(opt("--seed", 1))

POOLS = json.load(open(ERR_POOL if ERR_POOL.startswith("/") else f"{ROOT}/{ERR_POOL}")) if ERR_POOL else None
drop = Droplet(DROP_D)
t0 = time.time()
target, tv, tf, tinfo = load_benchy(f"{ROOT}/assets/benchy_download/3DBenchy.stl", scale=SCALE, res=0.06e-3,
                                    pole_tip=POLE_TIP)
if ZCUT > -0.5:
    zax = target.lo[2] + target.res * np.arange(target.shape[2])
    target.sdf = np.maximum(target.sdf, (ZCUT - zax)[None, None, :].astype(np.float32))
print(f"target: Benchy x{SCALE}, {np.round(tinfo['size']*1e3,1)} mm, {target.volume()*1e9:.0f} mm3 ({time.time()-t0:.0f}s)")


def describe(d, th):
    r = cap_sphere_radius(d.a, th)
    return (f"{2*d.a*1e3:.2f} mm ({d.vol*1e12:.1f} nL) -> contact angle {np.degrees(th):.0f} deg, cap height "
            f"{r*(1-np.cos(th))*1e3:.3f} mm, footprint {2*r*np.sin(th)*1e3:.3f} mm")


theta_c = cure_contact_angle(drop, DELAY)
print(f"viscosity {drop.mu*1e3:.0f} mPa.s; UV flash {DELAY*1e3:.0f} ms after touchdown; droplet {describe(drop, theta_c)}")
# per-class droplet properties: class 0 = normal droplet, class 1 = fine shell droplet
CLS = [dict(drop=drop, theta=theta_c, vd=drop.vol * 0.95)]
if SKIN == "fine":
    drop_s = Droplet(SKIN_D)
    theta_s = cure_contact_angle(drop_s, DELAY)
    CLS.append(dict(drop=drop_s, theta=theta_s, vd=drop_s.vol * 0.95))
    s_core = np.sqrt(CLS[0]["vd"] / (np.sqrt(3) / 2 * LAYER_H))
    if EDGES_ONLY:
        # small droplets only where the target has a sharp edge or corner: within EDGE_R of an edge and within SHELL of the surface
        sys.path.insert(0, f"{ROOT}/studies")
        from s15_edges import classify
        from scipy.ndimage import distance_transform_edt
        ep, _, eflag = classify(target)
        em = np.ones(target.shape, bool)
        ei = np.clip(np.round((ep[eflag] - target.lo) / target.res).astype(int), 0, np.array(target.shape) - 1)
        em[ei[:, 0], ei[:, 1], ei[:, 2]] = False
        d_edge = (distance_transform_edt(em) * target.res).astype(np.float32)
        near = lambda p: map_coordinates(d_edge, ((p - target.lo) / target.res).T, order=1, mode="nearest") < EDGE_R
        zone = lambda p: near(p) & (target.dist(p) > -SHELL)
        core = Lattice(target, drop, LAYER_H, POLE_TIP[2], inset=0.0, mask=lambda p: ~zone(p))
        fine = Lattice(target, drop_s, LAYER_H * SKIN_D / DROP_D, POLE_TIP[2], inset=0.0, mask=zone)
    else:
        core = Lattice(target, drop, LAYER_H, POLE_TIP[2], inset=SHELL / s_core)
        fine = Lattice(target, drop_s, LAYER_H * SKIN_D / DROP_D, POLE_TIP[2], inset=0.0, shell=SHELL)
    lat = MixedLattice(core, fine, POLE_TIP[2])
    cls = lat.cls
    h_site = lat.h
    print(f"shell droplet {describe(drop_s, theta_s)}")
    print(f"core lattice: {core.n} sites (pitch {core.s*1e3:.3f} mm, layer {core.h*1e3:.2f} mm); "
          f"shell lattice: {fine.n} sites (pitch {fine.s*1e3:.3f} mm, layer {fine.h*1e3:.2f} mm, shell {SHELL*1e6:.0f} um deep)")
else:
    lat = Lattice(target, drop, LAYER_H, POLE_TIP[2], inset=opt("--inset", 0.0))
    cls = np.zeros(lat.n, np.int8)
    h_site = np.full(lat.n, LAYER_H)
depth = POLE_TIP[2] - lat.pts[:, 2]
seeds = np.where((depth < 2 * LAYER_H) & (np.hypot(lat.pts[:, 0], lat.pts[:, 1]) < POLE_RADIUS + 0.5 * lat.s))[0]
if len(seeds) == 0:
    seeds = np.argsort(np.linalg.norm(lat.pts - POLE_TIP, axis=1))[:6]
order, done = lat.order(seeds)
print(f"lattice: {lat.n} sites, {len(seeds)} seeds on the pole face; connected to the pole: {100*done.mean():.2f} %")

lo, hi = target.bounds()
part = PrintedPart(lo - 1.2e-3, np.array([hi[0] + 1.2e-3, hi[1] + 1.2e-3, POLE_TIP[2] + 2e-3]), res=RES, band=0.5e-3)
part.owner = np.zeros(part.shape, np.int32)     # which droplet made each voxel solid (for the viewer)
part.add_cylinder(POLE_TIP, POLE_TIP + np.array([0, 0, 3e-3]), POLE_RADIUS)

N = lat.n
OUT = f"{ROOT}/out/print{TAG}"
os.makedirs(OUT, exist_ok=True)
bc = 0.5 * (lat.pts.min(0) + lat.pts.max(0))
br = float(np.linalg.norm(lat.pts - bc, axis=1).max())
meta = dict(n_sites=int(N), drop_d=DROP_D, layer_h=LAYER_H, pitch=float(lat.s), theta_c_deg=float(np.degrees(theta_c)),
            pole_tip=POLE_TIP.tolist(), bound_center=bc.tolist(), bound_radius=br, scale=SCALE,
            sigma_um=SIGMA * 1e6, p_fail=P_FAIL, viscosity_mPas=drop.mu * 1e3, delay_ms=DELAY * 1e3, surface=None, err_pool=bool(POOLS),
            record_floats=10, done=False, n_written=0, skin=SKIN, skin_d=SKIN_D if SKIN == "fine" else None,
            shell_um=SHELL * 1e6 if SKIN == "fine" else None)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
fdep = open(f"{OUT}/deposits.bin", "wb")
checkpoints = {int(f * N): f for f in (0.003, 0.03, 0.10, 0.25, 0.45, 0.65, 0.85, 0.99)}
stats = dict(blocked=0, tilted=0, lost=0, waited=0, up=0, side=0, down=0, attempts=0, fine=0, coarse=0)

# ---- closed-loop print: a site is attempted only when real material is next to it;
#      lost droplets are detected (camera) and re-fired; sites with no surface nearby
#      or no clear approach wait until a neighbour has actually been printed.
deposited = np.zeros(N, bool); queued = np.zeros(N, bool); tries = np.zeros(N, np.int16)
heap = []; cnt = 0
MAX_TRIES = 8


def push(i):
    global cnt
    heapq.heappush(heap, (int(lat.layer[i]), cnt, int(i))); queued[i] = True; cnt += 1


# Top-skin rule: a site with nothing above it inside the target (the upper skin of a
# roof, deck or ledge) is not hung sideways off a thin rim.  It waits until the layer
# beneath it exists and is then placed from above, so the liquid lands on a surface
# and wicks into the gaps between the beads below.
TOPSKIN = "--no_topskin" not in args
zs = lat.pts[:, 2]
lower = [[j for j in lat.nbrs[i] if zs[j] < zs[i] - 0.4 * h_site[i]] for i in range(N)]
above = lat.pts.copy(); above[:, 2] += h_site
is_top = (target.dist(above) > 0) & np.array([len(l) >= 2 for l in lower]) & TOPSKIN
is_top[seeds] = False
print(f"top-skin sites placed from above: {int(is_top.sum())} ({100*is_top.mean():.1f} %)")

for sidx in seeds:
    push(sidx)
t0 = time.time()
while heap:
    _, _, i = heapq.heappop(heap)
    queued[i] = False
    if deposited[i] or tries[i] >= MAX_TRIES:
        continue
    if is_top[i] and sum(deposited[j] for j in lower[i]) < min(2, len(lower[i])):
        continue                                  # re-queued when a lower neighbour is printed
    tries[i] += 1
    stats["attempts"] += 1
    k = CLS[cls[i]]
    a_i = k["drop"].a
    contact, n, ok = part.surface_point(lat.pts[i])
    if not ok:
        stats["waited"] += 1
        continue                                  # re-queued when a neighbour is printed
    if not part.approach_clear(contact, n, a_i):
        if any(part.approach_clear(contact, dd, a_i) for dd in tilted_directions(n)[1:]):
            stats["tilted"] += 1
        else:
            stats["blocked"] += 1
            continue                              # cannot be reached now; may clear up later
    pool = None
    if POOLS:
        kind = "benchy_down" if n[2] < -0.5 else "benchy_side" if n[2] <= 0.5 else "smooth_other"
        pool = POOLS.get(f"{int(round(2e6 * a_i))}_{kind}") or POOLS.get(f"300_{kind}")
    if rng.random() < (pool["lost"] / pool["n"] if pool else P_FAIL):
        stats["lost"] += 1
        push(i)                                   # miss detected -> fire again
        continue
    fr = frame_from_normal(n)
    if pool:
        mag, ph = rng.choice(pool["err_um"]) * 1e-6, rng.uniform(0, 2 * np.pi)
        e = mag * np.array([np.cos(ph), np.sin(ph)])
    else:
        e = rng.normal(0, SKIN_SIGMA if cls[i] else SIGMA, 2)
    c_land, n_land, ok2 = part.surface_point(contact + e[0] * fr[0] + e[1] * fr[1])
    if not ok2:
        c_land, n_land = contact, n
    if MODEL == "capillary":
        cb, rb = part.deposit_capillary(c_land, n_land, k["theta"], k["vd"], a_i, half=4.0 * a_i)
    else:
        cb, rb = part.deposit(c_land, n_land, k["theta"], k["vd"], a_i)
    deposited[i] = True
    stats["up" if n[2] > 0.5 else "down" if n[2] < -0.5 else "side"] += 1
    stats["fine" if cls[i] else "coarse"] += 1
    fdep.write(np.concatenate([cb, [rb], c_land, n_land]).astype("<f4").tobytes())
    for j in lat.nbrs[i]:
        if not deposited[j] and not queued[j] and tries[j] < MAX_TRIES:
            push(j)
    nd = part.n_deposits
    if nd % 500 == 0:
        fdep.flush(); meta["n_written"] = int(nd); json.dump(meta, open(f"{OUT}/meta.json", "w"))
    if nd in checkpoints:
        cand = []
        for (_, _, j) in sorted(heap)[:600:10]:
            c2, n2, okc = part.surface_point(lat.pts[j])
            if okc and part.approach_clear(c2, n2, CLS[cls[j]]["drop"].a):
                cand.append(np.concatenate([c2, n2]))
        np.savez_compressed(f"{OUT}/snap_{checkpoints[nd]:.3f}.npz", sdf=part.sdf.astype(np.float16),
                            lo=part.lo, res=part.res, n_dep=nd, cand=np.array(cand))
        print(f"  [{100*nd/N:5.1f} %] {nd} droplets, {time.time()-t0:.0f}s, snapshot saved ({len(cand)} candidate sites)", flush=True)
fdep.close()
meta.update(n_written=int(part.n_deposits), done=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
stats["never_printed"] = int((~deposited).sum())
print(f"deposited {part.n_deposits} of {N} sites in {time.time()-t0:.0f}s; stats {stats}")

# ---- fidelity metrics ------------------------------------------------------
ax = [target.lo[d] + target.res * np.arange(target.shape[d]) for d in range(3)]
X, Y, Z = np.meshgrid(*ax, indexing="ij")
surf = np.abs(target.sdf) < 0.75 * target.res
below_pole = Z[surf] < POLE_TIP[2] - 0.05e-3
pts = np.stack([X[surf], Y[surf], Z[surf]], 1)[below_pole]
pts = pts[pts[:, 2] > ZCUT + 0.15e-3]
gt = np.stack([target.dist(pts + e) - target.dist(pts - e) for e in np.eye(3) * target.res], 1)
pts = pts - target.dist(pts)[:, None] * gt / np.maximum(np.linalg.norm(gt, axis=1), 1e-12)[:, None]     # onto the target surface itself
tsdf = true_sdf(part.sdf, part.res)              # distance to the outer printed surface, valid deep inside too
dev = map_coordinates(tsdf, ((pts - part.lo) / part.res).T, order=1, mode="nearest")   # +: print is short, -: proud
dev_clip = np.clip(dev, -part.band, part.band)
inner = target.sdf < -0.2e-3
ipts = np.stack([X[inner], Y[inner], Z[inner]], 1)
ipts = ipts[ipts[:, 2] > ZCUT + 0.3e-3]
void = float((part.dist(ipts[::7]) > 0).mean())
res = dict(
    droplets=int(part.n_deposits), sites=int(N), stats=stats, skin_mode=SKIN, err_pool=ERR_POOL,
    skin_d_mm=SKIN_D * 1e3 if SKIN == "fine" else None, shell_um=SHELL * 1e6 if SKIN == "fine" else None,
    skin_sigma_um=SKIN_SIGMA * 1e6 if SKIN == "fine" else None,
    completeness=float(deposited.mean()),
    target_volume_mm3=target.volume() * 1e9, printed_volume_mm3=(part.volume() - np.pi * POLE_RADIUS ** 2 * 2e-3) * 1e9,
    surf_dev_um=dict(mean_signed=float(dev_clip.mean() * 1e6), mean_abs=float(np.abs(dev_clip).mean() * 1e6), rms=float(np.sqrt((dev_clip ** 2).mean()) * 1e6),
                     p95=float(np.percentile(np.abs(dev_clip), 95) * 1e6), p99=float(np.percentile(np.abs(dev_clip), 99) * 1e6),
                     frac_over_300um=float((np.abs(dev_clip) > 0.3e-3).mean())),
    interior_void_fraction=void, sigma_um=SIGMA * 1e6, p_fail=P_FAIL, drop_d_mm=DROP_D * 1e3,
    pitch_mm=lat.s * 1e3, layer_mm=LAYER_H * 1e3, theta_c_deg=float(np.degrees(theta_c)),
    normals=dict(up=stats["up"], side=stats["side"], down=stats["down"]))
print(json.dumps(res, indent=1))
json.dump(res, open(f"{ROOT}/out/print{TAG}/result.json", "w"), indent=1)

# ---- cured-surface mesh for the viewer: every vertex knows which droplet completed it
from skimage.measure import marching_cubes
from scipy.ndimage import maximum_filter
import fast_simplification as fs
f = np.pad(part.sdf.astype(np.float32), 1, mode="constant", constant_values=part.band)
v, faces, _, _ = marching_cubes(f, level=0.0)
v = (v - 1.0) * part.res + part.lo
keep = v[faces].mean(1)[:, 2] < POLE_TIP[2] + 0.05e-3            # drop the pole stub
faces = faces[keep]
v2, f2 = fs.simplify(v.astype(np.float32), faces.astype(np.int32), target_count=min(len(faces), 420000), agg=4)
own = maximum_filter(part.owner, size=3)
nrm = part.normal(v2.astype(float))
ii = np.clip(np.round((v2 - nrm * 0.03e-3 - part.lo) / part.res).astype(int), 0, np.array(part.shape) - 1)
vt = own[ii[:, 0], ii[:, 1], ii[:, 2]].astype(np.float32)
v2.astype("<f4").tofile(f"{OUT}/surface_pos.bin"); f2.astype("<u4").tofile(f"{OUT}/surface_idx.bin"); vt.astype("<f4").tofile(f"{OUT}/surface_time.bin")
meta.update(surface=dict(n_verts=int(len(v2)), n_faces=int(len(f2))), model=MODEL)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print(f"surface mesh: {len(v2)} vertices, {len(f2)} faces")
np.savez_compressed(f"{ROOT}/out/print{TAG}/final.npz", sdf=part.sdf.astype(np.float16), lo=part.lo, res=part.res)
