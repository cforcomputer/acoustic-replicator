import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from replicator import configs
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field, frame_from_normal
from replicator.holo import TrapSolver
drop = Droplet(0.3e-3)
arr = configs.REGISTRY["box6_12"]()
gaps = np.array([0.03, 0.3, 0.6, 0.9, 1.2]) * 1e-3
for scene, R, ext, n in (("tip", 1.5e-3, 8e-3, [0, 0, -1]), ("tip", 1.5e-3, 8e-3, [1, 0, 0]), ("sph20", 10e-3, 16e-3, [0, 0, -1])):
    n = np.array(n, float)
    part = new_part(extent=ext, pole_tip_z=max(R - 2e-3, 0.0), res=0.2e-3 if R > 6e-3 else 0.1e-3)
    part.add_sphere([0, 0, 0], R)
    xs = part.project((n * R)[None])[0]
    fr = frame_from_normal(part.normal(xs[None])[0])
    fld = build_field(part, arr, target=xs)
    sv = TrapSolver(fld, drop)
    t0 = time.time()
    phi, s, info = sv.solve_funnel(xs, fr, gaps)
    print(f"{scene} n={n}: [{info['init']}] {time.time()-t0:.0f}s  f_n={np.round(info['f_n'],2)}  |f_t|={np.round(info['f_t'],3)}  "
          f"kappa_t={np.round(info['kappa_t'],2)}  Ba={info['Ba']:.2f} amp={s:.2f}", flush=True)
