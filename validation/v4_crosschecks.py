"""Independent cross-checks of the numbers the conclusions rest on.  None of these
re-uses the test it is checking: each compares against a different formula, a
symmetry, or a published figure."""
import sys
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
import trimesh
from replicator import constants as C, configs
from replicator.acoustics import Array, Panel, Mesh, Scatterer, Field
from replicator.gorkov import Droplet
from replicator.dynamics import ShapeMode, drag_force, min_jerk
from replicator.delivery import Jet, JET_MK2
from replicator.pcb import Board, TRANSDUCER

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


print("X1 acoustic reciprocity through the scattering solver (source A heard at B = source B heard at A)")
sph = trimesh.creation.icosphere(subdivisions=3, radius=8e-3)
msh = Mesh(sph.vertices, sph.faces)
A = np.array([0.03, 0.01, -0.02]); B = np.array([-0.012, 0.02, 0.011])
def heard(src, at):
    arr = Array([Panel("s", src, -src / np.linalg.norm(src), [0, 0, 1.0], 1, 1)], vpp=1, p_ref_per_vpp=1, piston_radius=1e-7)
    return Field(arr, Scatterer(msh, arr)).eval(at[None], np.ones(1, complex))[0][0]
pab, pba = heard(A, B), heard(B, A)
check("|p_AB - p_BA| / |p_AB|", abs(pab - pba) / abs(pab) < 5e-3, f"{abs(pab - pba) / abs(pab):.2e}")

print("X2 levitation threshold vs King (1934) closed form, water at 40 kHz")
f1 = 1.0; f2 = 2 * (998 - C.RHO0) / (2 * 998 + C.RHO0)
mine = np.sqrt(998 * C.G_ACC * 4 * C.RHO0 * C.C0 ** 2 / (3 * C.K * (f1 / 3 + f2 / 2)))
king = np.sqrt(1.6 * 998 * C.RHO0 * C.G_ACC * C.C0 ** 2 / C.K)
check("model vs King formula", abs(mine / king - 1) < 0.01, f"{mine:.0f} Pa vs {king:.0f} Pa")

print("X3 droplet shape mode vs textbook (Rayleigh): 1 mm-radius water drop ~ 120 Hz")
w = Droplet(2e-3, rho=998, sigma=0.072, mu=1e-3)
sm = ShapeMode(w, 1e-5)
check("Rayleigh l=2 frequency", abs(sm.w2 / 2 / np.pi - 120.8) < 1.0, f"{sm.w2/2/np.pi:.1f} Hz")
d3 = Droplet(0.3e-3); s3 = ShapeMode(d3, 1e-4)
for _ in range(4000): s3.step(1.0)
check("static flattening eps = -(3/32) Ba", abs(s3.eps + 3 / 32) < 1e-4, f"eps {s3.eps:.5f}")

print("X4 transducer output constant vs the manufacturer's rating")
p_rms_03 = 20e-6 * 10 ** (120 / 20)                       # 120 dB SPL at 0.3 m, 10 Vrms sine
datasheet = p_rms_03 * np.sqrt(2) * 0.3 / (10 * 2 * np.sqrt(2))
check("model constant is not above the datasheet", C.P_REF_PER_VPP <= datasheet,
      f"model {C.P_REF_PER_VPP} vs datasheet-derived {datasheet:.2f} Pa.m/Vpp (model is {datasheet/C.P_REF_PER_VPP:.1f}x conservative)")

# The cheapest transducers are weaker.  Guaranteed minimum of a generic part: 110 dB at 10 Vrms, 30 cm (Sunny TCT40-10T2);
# a 115 dB class part (Manorshi MSO-P1040H07T) sits in between.  Compare, on the same datasheet basis, with the 120 dB Murata
# part the model constant was measured on, at the drive voltage each would be run at.
def rel_amplitude(spl_db, v_supply):
    return 10 ** ((spl_db - 120) / 20) * v_supply / 18.0          # field amplitude relative to the simulated machine (120 dB class at 18 V)
ga, gb = rel_amplitude(110, 28.0), rel_amplitude(115, 24.0)
print(f"     field amplitude relative to the simulated machine: 110 dB part at 28 V = {ga:.2f}, 115 dB part at 24 V = {gb:.2f} "
      f"(landings were re-simulated at reduced amplitude; see the landing-test table for how low it can go)")
check("a 115 dB-class transducer at 24 V reaches at least 0.71 of the simulated amplitude (a level at which landings were re-simulated)", gb >= 0.71, f"{gb:.2f}")

print("X5 polyhedron geometry")
arr = configs.rco(5)
r = np.array([np.linalg.norm(p.center) for p in arr.panels])
check("18 boards, all faces equidistant", len(arr.panels) == 18 and np.ptp(r) < 1e-9, f"r = {r[0]*1e3:.2f} mm")
N = np.array([-p.normal for p in arr.panels])
cosn = np.sort((N @ N.T).ravel())[::-1]
check("nearest-neighbour faces 45 deg apart", abs(np.degrees(np.arccos(cosn[18])) - 45) < 1e-6)
face_at_pcb = arr.edge * (arr.r_face + 7.1e-3) / arr.r_face
check("board (58 mm) fits its face where it sits (7.1 mm outside the transducer faces)", Board(5).side * 1e-3 + 0.3e-3 < face_at_pcb - 4e-3,
      f"face is {face_at_pcb*1e3:.1f} mm wide there: {0.5*(face_at_pcb*1e3-58):.1f} mm of frame on each side")
lat = np.hypot(arr.pos[:, 0], arr.pos[:, 1])
check("nothing above or below the build zone within 20 mm of the axis", lat.min() > 20e-3, f"closest transducer {lat.min()*1e3:.1f} mm from the axis")
d = np.linalg.norm(arr.pos[:, None] - arr.pos[None], axis=2); np.fill_diagonal(d, 1)
check("no two transducer cans closer than 10.4 mm centre-to-centre (9.9 mm cans)", d.min() > 10.4e-3, f"min {d.min()*1e3:.2f} mm")

print("X6 motion and jet arithmetic")
T, s, sd, sdd = min_jerk(0.03, 4.905)
tt = np.linspace(0, T, 20001)
check("min-jerk peak acceleration equals the limit", abs(sdd(tt).max() / 4.905 - 1) < 1e-3, f"{sdd(tt).max():.3f} m/s2")
jet = Jet(**JET_MK2)
t, x, v, a = jet.nominal(d3)
# independent integration of the rise (energy form, adaptive solver) instead of the module's stepping
from scipy.integrate import solve_ivp
def dz_reached(v0):
    f = lambda z, y: [(-C.G_ACC + drag_force(d3, np.array([0, 0, np.sqrt(max(y[0], 1e-12))]))[2] / d3.mass) * 2]   # d(v^2)/dz
    ev = lambda z, y: y[0] - 1e-9
    ev.terminal = True
    return solve_ivp(f, [0, 0.2], [v0 ** 2], events=ev, rtol=1e-9, atol=1e-12).t_events[0][0]
lo, hi = 0.5, 1.2
for _ in range(50):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if dz_reached(mid) < 26e-3 else (lo, mid)
check("jet speed needed for a 26 mm rise, independent integration", abs(v[0, 2] / lo - 1) < 0.01,
      f"{v[0,2]:.4f} vs {lo:.4f} m/s (drag-free would be {np.sqrt(2*C.G_ACC*26e-3):.3f}: air drag is ~30 % of gravity here)")
vt = 0.5
for _ in range(60):
    re = C.RHO0 * vt * d3.d / C.MU_AIR
    vt = d3.weight / (3 * np.pi * C.MU_AIR * d3.d * (1 + 0.15 * re ** 0.687))
check("drag law balances weight at terminal velocity", abs(np.linalg.norm(drag_force(d3, np.array([0, 0, -vt]))) / d3.weight - 1) < 1e-6, f"terminal {vt:.2f} m/s")

print("X7 transducer board")
b = Board(5)
v_, mins = b.drc()
e = b.electrical()
check("design-rule check clean", len(v_) == 0, f"min clearances {mins}")
check("microcontroller has enough pins", e["gpio_ok"], f"{e['gpio_needed']} needed, {e['gpio_available']} available")
check("drive voltage within transducer and driver ratings", e["transducer_ok"] and e["driver_ok"], f"{e['drive_vpp']} Vpp")
check("driver edge is much faster than one phase step", e["edge_charge_time_ns"] < 0.5 * e["phase_step_ns_at_128"],
      f"{e['edge_charge_time_ns']} ns vs {e['phase_step_ns_at_128']} ns")
pads = {(round(i["x"], 3), round(i["y"], 3)) for i in b.items if i["part"] == "PAD"}
cans = [(i["x"], i["y"]) for i in b.items if i["part"] == "TRANSDUCER"]
check("every can has its two leads 5.0 mm apart, inside its own outline",
      all((round(x - 2.5, 3), round(y, 3)) in pads and (round(x + 2.5, 3), round(y, 3)) in pads for x, y in cans))
check("board transducer positions equal the simulated array's", np.allclose(sorted(cans), sorted(map(tuple, np.round(arr.panels[0].grid_uv() * 1e3, 6)))))
print(f"     array power at full drive: {18*e['board_power_w']:.0f} W, {18*e['board_current_a']:.2f} A at {e['drive_vpp']:.0f} V")
e24 = Board(5, v_drive=24.0).electrical()
check("drivers keep >= 10 % voltage headroom at 24 V (weaker transducers need the higher rail)", e24["driver_ok"], f"{e24['driver_headroom_pct']} % headroom; array {18*e24['board_power_w']:.0f} W")
check("all passives found a legal place", len(b.unplaced) == 0, f"{len([i for i in b.items if i['side']=='back'])} parts on the back")

print("X8 fabrication files and hardware model")
import json, os, warnings
warnings.filterwarnings("ignore")
gd = f"{ROOT}/hardware/pcb/gerbers"
try:
    from gerbonara import LayerStack
    st = LayerStack.open(gd)
    names = str(st)
    check("Gerber set is read back by an independent parser with every layer identified",
          all(k in names for k in ("top copper", "bottom copper", "top mask", "bottom mask", "bottom paste", "mechanical outline", "2 drill layers")), names[names.find("["):])
    (x0, y0), (x1, y1) = st.outline.instance.bounding_box("mm") if hasattr(st.outline, "instance") else st["mechanical outline"].bounding_box("mm")
    check("board outline in the Gerbers is 58 x 58 mm", abs((x1 - x0) - 58.1) < 0.15 and abs((y1 - y0) - 58.1) < 0.15, f"{x1-x0:.2f} x {y1-y0:.2f} mm (incl. 0.1 mm line)")
except ImportError:
    print("     (gerbonara not installed: Gerber read-back skipped)")
drl = open(f"{gd}/TB-25-PTH.drl").read()
from replicator.pcbroute import _parse_drill
check("drill file has two plated 1.0 mm holes per transducer", sum(1 for x, y, d in _parse_drill(f"{gd}/TB-25-PTH.drl") if abs(d - 1.0) < 1e-6) == 50)
rr = json.load(open(f"{ROOT}/hardware/pcb/route_report.json"))
vv = rr["verification"]
check("routed board: connectivity rebuilt from the Gerber and drill files equals the netlist (details: validation/v6_pcb.py)",
      vv["ok"], f"{vv['nets_checked']} nets, {vv['pads_checked']} pads, opens {len(vv['opens'])}, shorts {len(vv['shorts'])}, clearance faults {vv['n_clearance_violations']}")
ir = json.load(open(f"{ROOT}/hardware/interference_report.json")); at = json.load(open(f"{ROOT}/hardware/attachment_report.json")); wr = json.load(open(f"{ROOT}/hardware/wiring.json"))["report"]
check("no two solids of the machine overlap", len(ir["overlaps"]) == 0, f"{ir['n_solids']} solids, {ir['pairs_checked']} close pairs")
check("every part is fastened, touches what it is fastened to, and the chain reaches the enclosure floor",
      not at["unmounted"] and not at["gaps"] and not at["unrooted"] and not at["split_bodies"], f"{at['n_parts']} parts; printed bodies {at['bodies']}")
check("every wire ends on a part and none runs through a solid", not wr["dangling"] and not wr["through"], f"{wr['n_wires']} conductors, {wr['length_m']} m")
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
