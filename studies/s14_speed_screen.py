"""Study 14 (final design, static checks):
 A. Several droplets at once: one hologram holding 1, 2, 3, 4 traps along the ascent line.
 B. Protective mesh over the six lower boards: trap and funnel quality vs mesh transmission."""
import sys, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import configs, constants as C
from replicator.acoustics import Field
from replicator.gorkov import Droplet, TrapStencil
from replicator.holo import TrapSolver, Term
from replicator.scene import new_part, build_field, frame_from_normal

drop = Droplet(0.3e-3)
out = {}
print("A. simultaneous traps (free space, 18 Vpp, integrity limit Ba <= 1 at every droplet)")
arr = configs.REGISTRY["rco5"]()
fld = Field(arr)
sv = TrapSolver(fld, drop)
sets = {1: [[0, 0, -12e-3]], 2: [[0, 0, -12e-3], [0, 0, -34e-3]], 3: [[0, 0, -12e-3], [0, 0, -27e-3], [0, 0, -42e-3]],
        4: [[6e-3, 0, -12e-3], [0, 0, -24e-3], [0, 0, -36e-3], [0, 0, -48e-3]]}
rng = np.random.default_rng(0)
out["multi"] = {}
for nt, pts in sets.items():
    terms = [Term(TrapStencil(fld, np.array(p), drop, sv.h), np.eye(3), np.zeros(3), 1.0 / nt) for p in pts]
    best = None
    for trial in range(4):
        phi0 = rng.uniform(0, 2 * np.pi, arr.n) if trial else arr.focus_phases(np.array(pts[0]))
        phi, s, J, nit = sv._run(terms, phi0, 0.7, 600)
        dg = [sv.diagnose(t.st, phi, s, np.eye(3), np.zeros(3)) for t in terms]
        kmin = min(d["kappa"].min() for d in dg)
        if best is None or kmin > best[0]:
            best = (kmin, dg, s)
    kmin, dg, s = best
    print(f"  {nt} trap(s): weakest stiffness {kmin:.2f} weights/mm (per trap: {[round(float(d['kappa'].min()), 2) for d in dg]}), "
          f"Bond {[round(d['Ba'], 2) for d in dg]}, residual force {max(np.linalg.norm(d['f_res']) for d in dg):.3f} W, drive {s:.2f}")
    out["multi"][nt] = dict(kappa_min=float(kmin), per=[float(d["kappa"].min()) for d in dg])

print("B. acoustic mesh over the six lower boards (amplitude transmission T)")
R = 10e-3
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
xs = part.project(np.array([[0, 0, -R]]))[0]
fr = frame_from_normal(part.normal(xs[None])[0])
out["mesh"] = {}
for T in (1.0, 0.92, 0.84, 0.7, 0.5):
    a2 = configs.REGISTRY["rco5"]()
    for i, p in enumerate(a2.panels):
        if p.normal[2] > 0.5:                       # boards below the build zone face upward
            a2.gain[a2.panel_id == i] = T
    f0 = Field(a2); s0 = TrapSolver(f0, drop)
    k_free = min(float(s0.solve_best(np.array(x), n_random=1)[2]["kappa"].min()) for x in ([0, 0, -14e-3], [10e-3, 0, -4e-3], [0, 0, -34e-3]))
    f2 = build_field(part, a2, target=xs); s2 = TrapSolver(f2, drop)
    phi, s, info = s2.solve_funnel(xs, fr, np.array([0.03, 0.3, 0.6, 0.9, 1.2]) * 1e-3, n_random=2)
    print(f"  T = {T:.2f} ({20*np.log10(T):+.1f} dB): free-space trap {k_free:.2f} weights/mm, underside funnel centring {info['kappa_t'].min():.2f}, "
          f"push {info['f_n'].max():+.2f}..{info['f_n'].min():+.2f} W", flush=True)
    out["mesh"][T] = dict(k_free=k_free, funnel_kt=float(info["kappa_t"].min()))
json.dump(out, open(f"{ROOT}/out/speed_screen.json", "w"), indent=1)
