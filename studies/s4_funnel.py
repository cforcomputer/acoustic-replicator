"""Study 4: static 'funnel' hologram for the final approach.
For each site: solve one hologram, then map the net force over the whole last
1.5 mm (normal and tangential offsets) to confirm it is inward and centring everywhere."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from replicator import configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field, frame_from_normal
from replicator.holo import TrapSolver, force_at

args = sys.argv[1:]
cfg = next((a for a in args if a in configs.REGISTRY), "box6_12")
R = 10e-3
drop = Droplet(0.3e-3)
arr = configs.REGISTRY[cfg]()
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
DIRS = {"bottom": [0, 0, -1], "lower45": [1, 0, -1], "side": [1, 0, 0], "upper45": [1, 0, 1]}
sel = [a for a in args if a in DIRS] or list(DIRS)
push = float(args[args.index("--push") + 1]) if "--push" in args else 1.0
gaps = np.array([0.03, 0.3, 0.6, 0.9, 1.2]) * 1e-3
for sname in sel:
    n = np.array(DIRS[sname], float); n /= np.linalg.norm(n)
    xs = part.project((n * R)[None])[0]
    fr = frame_from_normal(part.normal(xs[None])[0])
    fld = build_field(part, arr, target=xs)
    sv = TrapSolver(fld, drop)
    t0 = time.time()
    phi, s, info = sv.solve_funnel(xs, fr, gaps, push=push)
    print(f"== {cfg} {sname}: funnel solved in {time.time()-t0:.0f}s [{info['init']}], amp {s:.2f}, Ba {info['Ba']:.2f}")
    print(f"   at design gaps: f_n = {np.round(info['f_n'],2)}  |f_t| = {np.round(info['f_t'],3)}  kappa_t = {np.round(info['kappa_t'],2)}")
    w = s * np.exp(1j * phi)
    g = np.linspace(0.02e-3, 1.6e-3, 33)
    worst_fn, worst_ct = -np.inf, np.inf
    for off in (0.0, 0.1e-3, 0.2e-3):
        for td in ((fr[0], fr[1], -fr[0], -fr[1]) if off > 0 else (fr[0],)):
            pts = xs + fr[2] * (drop.a + g)[:, None] + td * off
            f = force_at(fld, drop, pts, w)
            fn = f @ fr[2]
            ft = f @ td                      # +ve = pushed further off-axis
            worst_fn = max(worst_fn, fn[g <= 1.25e-3].max())
            if off > 0:
                worst_ct = min(worst_ct, (-ft[g <= 1.25e-3]).min())
            if off == 0.0:
                print("   on-axis f_n vs gap(mm):", " ".join(f"{gg*1e3:.2f}:{v:+.2f}" for gg, v in zip(g[::4], fn[::4])))
    print(f"   worst case over gap<=1.25mm, |offset|<=0.2mm: max f_n = {worst_fn:+.2f} W (must be <0); "
          f"min centring force = {worst_ct:+.2f} W (must be >0)", flush=True)
