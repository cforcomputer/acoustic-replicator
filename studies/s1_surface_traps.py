"""Study 1: can a stable trap be formed at / near the surface of a hanging part?

Part: 20 mm sphere on the pole.  For several landing sites and stand-off
distances, solve the scattering-aware hologram and report trap quality.
Usage: s1_surface_traps.py [config ...]
"""
import sys, time, json
import numpy as np

sys.path.insert(0, ".")
from replicator import constants as C
from replicator import configs
from replicator.gorkov import Droplet, TrapStencil
from replicator.holo import TrapSolver, restoring_profile, holding_force
from replicator.scene import new_part, build_field, frame_from_normal

R = 10e-3
DROP_D = float(sys.argv[sys.argv.index("--d") + 1]) * 1e-3 if "--d" in sys.argv else 0.4e-3
BA = float(sys.argv[sys.argv.index("--ba") + 1]) if "--ba" in sys.argv else 0.5
names = [a for a in sys.argv[1:] if a in configs.REGISTRY] or ["opposed16", "box6_12"]

drop = Droplet(DROP_D)
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)

sites = {
    "bottom (normal -z)": np.array([0, 0, -1.0]),
    "lower 45deg": np.array([1, 0, -1.0]) / np.sqrt(2),
    "side (equator)": np.array([1.0, 0, 0]),
    "upper 45deg": np.array([1, 0, 1.0]) / np.sqrt(2),
}
standoffs_mm = [5.0, 3.0, 2.1, 1.2, 0.6, 0.3]

for name in names:
    arr = configs.REGISTRY[name]()
    print(f"\n=== {arr.name}: {arr.n} transducers, {arr.vpp:.0f} Vpp, droplet {DROP_D*1e3:.2f} mm, Ba<={BA} ===")
    for sname, n in sites.items():
        xs = n * R                                   # surface point
        t0 = time.time()
        fld = build_field(part, arr, target=xs)
        fr = frame_from_normal(n)
        print(f"-- site {sname}: {fld.scat.mesh.n} panels, BEM {time.time()-t0:.1f}s")
        solver = TrapSolver(fld, drop, ba_max=BA, h=0.05e-3)
        prev = None
        for so in standoffs_mm:
            x = xs + n * (drop.a + so * 1e-3)
            solver.set_mode(frame=fr)
            extra = {"prev": prev} if prev is not None else None
            phi, amp, info = solver.solve_best(x, extra=extra, n_random=1)
            prev = phi
            w = amp * np.exp(1j * phi)
            s, f = restoring_profile(fld, drop, x, w, fr, reach=min(3e-3, so * 1e-3 * 0.95), n=61)
            hold = [min(holding_force(s, f[i])) for i in range(3)]
            print(f"   gap {so:4.1f} mm  trap  kappa={np.array2string(info['kappa'], precision=2, floatmode='fixed')} "
                  f"hold(t1,t2,n)={np.round(hold,2)} W  Ba={info['Ba']:.2f} amp={amp:.2f} "
                  f"|f_res|={np.linalg.norm(info['f_res']):.3f} [{info['init']}]")
        # press mode at contact
        x = xs + n * (drop.a + 0.02e-3)
        for push in (0.5, 2.0):
            solver.set_mode(frame=fr, press=push)
            phi, amp, info = solver.solve_best(x, extra={"prev": prev}, n_random=1)
            fn = info["f_net"] @ n
            ft = np.linalg.norm(info["f_net"] - fn * n)
            print(f"   CONTACT press {push:.1f}W: tangential kappa={np.round(info['kappa_sub'],2)} "
                  f"f_normal={fn:+.2f} W  |f_tang|={ft:.3f} W  Ba={info['Ba']:.2f} amp={amp:.2f} [{info['init']}]")
