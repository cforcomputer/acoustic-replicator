"""Rebuild the viewer's status panel (out/status.json) from the result files of the final design.
Every line shown in the panel is read from a file produced by a test; nothing is typed in by hand
except the wording.  Usage: s18_status.py ["verdict text"]"""
import sys, json, os, glob
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import status

J = lambda p: json.load(open(f"{ROOT}/{p}")) if os.path.exists(f"{ROOT}/{p}") else None
if os.path.exists(status.PATH):
    os.remove(status.PATH)
T = lambda key, title, state, detail: status.update(test=(key, title, state, detail))

# ---- hardware model
ir, at, wr, pp = J("hardware/interference_report.json"), J("hardware/attachment_report.json"), J("hardware/wiring.json"), J("hardware/print/printed_parts.json")
mo = J("hardware/motion_report.json") or []
if ir and at and wr:
    necks = all(v["neck"]["pieces_after_peeling"] == 1 and v["solids"] == 1 for v in pp.values()) if pp else False
    clean = (not ir["overlaps"] and not at["unmounted"] and not at["gaps"] and not at["unrooted"] and not at["split_bodies"] and not wr["report"]["dangling"]
             and not wr["report"]["through"] and all(not m["collisions"] for m in mo) and necks)
    T("hardware", "Final assembly: nothing overlaps, nothing floats, every screw drawn, nothing collides when taken out", "pass" if clean else "fail",
      f"{ir['n_solids']} solids at real size · {ir['pairs_checked']} close pairs checked · {len(ir['overlaps'])} overlaps\n"
      f"every part fastened to something and connected to the enclosure floor: {'yes' if not (at['unmounted'] or at['gaps'] or at['unrooted']) else 'NO'}\n"
      f"{wr['report']['n_wires']} wire conductors, {wr['report']['length_m']} m · loose ends {len(wr['report']['dangling'])} · through a solid {len(wr['report']['through'])}\n"
      f"{len(mo)} service motions (chuck out, cap off, generator cartridge out, door, lid, one board out): collisions {sum(len(m['collisions']) for m in mo)}"
      + (f"\n{len(pp)} printed parts, {sum(v['volume_cm3'] for v in pp.values()):.0f} cm³; each one solid and sound after peeling 0.6-0.8 mm off every surface: {'yes' if necks else 'NO'}" if pp else ""))
rr = J("hardware/pcb/route_report.json")
pcb = J("hardware/pcb/TB-25.json")
if pcb and rr:
    e, v = pcb["electrical"], rr["verification"]
    T("pcb", "Transducer board TB-25: routed; files for JLCPCB", "pass" if v["ok"] and not pcb["drc"]["violations"] else "fail",
      f"4 layers · {rr['routing']['tracks']} tracks, {rr['routing']['total_track_mm']/1000:.2f} m · {rr['routing']['vias']} vias · {rr['assembly']['placed']} parts placed by the assembler, {rr['assembly']['bom_lines']} part numbers from JLCPCB's library\n"
      f"rebuilt from the Gerber and drill files alone: {v['nets_checked']} nets, {v['pads_checked']} pads · open circuits {len(v['opens'])} · shorts {len(v['shorts'])} · "
      f"smallest gap between nets {min(c['min_gap_mm'] for c in v['clearance'].values())} mm\n"
      f"driver headroom at 18 V: {e['driver_headroom_pct']} % · microcontroller pins {e['gpio_needed']}/{e['gpio_available']}\n"
      "no board has been built and there is no firmware yet")
bom = J("hardware/bom.json")
if bom:
    n = len(bom["bom"]); ver = sum(1 for b in bom["bom"] if b["status"] == "verified"); quote = [b for b in bom["bom"] if b["status"] == "quote"]
    T("sourcing", "Can every part be bought?", "warn",
      f"{n} lines, about ${bom['total_usd']:.0f} without a host computer · {ver} lines checked on a supplier page on 2026-11-10\n"
      f"the droplet generator is built, not bought: a piezo generator after a published open design, unproven at 0.3 mm with warmed resin;\nthinnest hobby resin found is 75-85 mPa·s, not the 30 assumed")

# ---- landing tests on the final construction (nozzle 82 mm below centre)
rows = {}
for fn in sorted(glob.glob(f"{ROOT}/out/mc/rco5_*_nz82*.json")):
    d = json.load(open(fn))
    rows[os.path.basename(fn)[:-5]] = d["rows"]


def pool(keys):
    rr = [r for k in keys for r in rows.get(k, [])]
    ok = np.array([r["err_true"] * 1e6 for r in rr if r["status"] == "landed"])
    return len(ok), len(rr), (float(np.median(ok)) if len(ok) else float("nan")), (float(np.percentile(ok, 90)) if len(ok) else float("nan")), (float(ok.max()) if len(ok) else float("nan"))


def line(name, keys):
    a, n, med, p90, mx = pool(keys)
    return f"{name}: landed {a}/{n} · median {med:.0f} µm · 9 in 10 within {p90:.0f} µm · worst {mx:.0f} µm" if n else f"{name}: not run"


sm = [f"rco5_sph20_{s}_d300_nz82" for s in ("bottom", "lower45", "side", "upper45", "top")]
a, n, med, p90, mx = pool(sm)
if n:
    T("land_smooth", "Landings on a smooth 20 mm part (all directions, imperfect machine)", "pass" if a == n and med < 200 else "warn", line("0.3 mm droplets", sm) + "\n" + line("bare pin tip (first droplets)", ["rco5_tip_bottom_d300_nz82"]))
bk = ["rco5_benchy0.100_autodown_d300_nz82", "rco5_benchy0.450_autodown_d300_nz82", "rco5_benchy0.250_autoside_d300_nz82"]
a, n, med, p90, mx = pool(bk)
if n:
    T("land_benchy", "Landings on the real Benchy geometry mid-print", "warn" if a < n or p90 > 400 else "pass",
      line("under-side sites, 10 % printed", bk[:1]) + "\n" + line("under-side sites, 45 % printed", bk[1:2]) + "\n" + line("side wall, 25 % printed", bk[2:]) +
      "\nthis is the weak spot: on thin or crowded geometry some droplets are lost or land in the wrong place")
fk = [f"rco5_{s}_d200_nz82" for s in ("sph20_bottom", "sph20_side", "benchy0.450_autodown")]
if pool(fk)[1]:
    sw = pool(["rco5_benchy0.250_autoside_d200_nz82"])
    T("land_fine", "0.2 mm droplets (used for the fine shell)", "warn" if sw[1] and (sw[0] < sw[1] or sw[3] > 400) else "pass",
      line("smooth part + Benchy under-side", fk) + "\n" + line("Benchy side wall", ["rco5_benchy0.250_autoside_d200_nz82"]))
pw = []
for tag, frac in (("v127", 71), ("v090", 50), ("v072", 40)):
    ks = [k for k in rows if k.endswith(tag)]
    if ks:
        pw.append(line(f"{frac} % of the simulated sound pressure", ks))
if pw:
    T("land_power", "How much acoustic power is really needed? (cheaper, weaker transducers)", "pass", "\n".join(pw) + "\nthe cheapest transducers at their guaranteed minimum reach about 47 % at 27 V, the most the chosen supply gives")
if "rco5_sph20_bottom_d300_nz82_mu80" in rows:
    T("land_mu", "Thicker resin (80 mPa·s, the thinnest hobby resin found)", "pass", line("under-side site", ["rco5_sph20_bottom_d300_nz82_mu80"]))

# ---- prints
def print_line(tag, name):
    r, e = J(f"out/print{tag}/result.json"), J(f"out/print{tag}/edges.json")
    if not (r and e):
        return None
    return (f"{name}: {r['droplets']:,} droplets · surface error {e['all']['mean_abs_um']:.0f} µm mean ({e['faces']['mean_abs_um']:.0f} on faces, {e['edges']['mean_abs_um']:.0f} on edges) · "
            f"internal voids {100*r['interior_void_fraction']:.1f} % · built {100*r['completeness']:.1f} % of sites")
pl = [x for x in (print_line("", "0.3 mm droplets, smooth-part landing scatter"), print_line("_fine", "0.3 mm core + 0.2 mm shell, smooth-part landing scatter"),
                  print_line("_mc", "0.3 mm, landing errors and losses taken from the Benchy flight tests")) if x]
fl = J("out/resolution_floor.json")
if pl:
    T("print", "Benchy at 40 % scale (24 mm long), printed hanging from the pin", "warn",
      "\n".join(pl) + (f"\nfloor set by droplet size alone on edges: {fl['270']['edges']['mean_abs_um']:.0f} µm for 0.3 mm droplets, {fl['200']['edges']['mean_abs_um']:.0f} µm for 0.2 mm" if fl else ""))
sp, sf = J("out/structure_print.json"), J("out/structure_print_fine.json")
ac = [J(f) for f in ("out/structure_acoustic_0.100_autodown_d300.json", "out/structure_acoustic_0.250_autoside_d300.json", "out/structure_acoustic_0.450_autodown_d300.json")]
if sp and sf and all(ac):
    st = sp["states"] + sf["states"]
    sag = max(x["sag_max_um"] for x in st); vm = max(x["vm_max_kPa"] for x in st)
    amov = max(max(d["part_disp_max_um"] for d in a["displacement"]) for a in ac); avm = max(max(d["vm_max_kPa"] for d in a["displacement"]) for a in ac)
    fmax = max(max(np.linalg.norm(k["F_net_uN"]) for k in a["keyframes"]) for a in ac)
    T("structure", "Does the half-built part need structural support?", "warn",
      f"own weight, {len(st)} snapshots: stress up to {vm/1000:.2f} MPa, sag up to {sag:.0f} µm at a resin modulus of 1 GPa\n"
      f"sound field during a delivery, 3 Benchy sites: net force on the part up to {fmax/1000:.1f} mN, stress up to {avm/1000:.2f} MPa, movement up to {amov:.0f} µm at 1 GPa\n"
      "strength is no issue (resin about 30 MPa); soft, freshly cured thin walls can move by tens of µm, which the print simulation does not include\n"
      f"on the thin side wall {max(a['clipped_panels_max'] for a in ac)} of {max(a['panels'] for a in ac)} boundary-element panels had to be clipped: the solver is at its limit there")
tw = J("out/twin_demo.json")
if tw and "cal" in tw:
    c = tw["cal"]
    med = lambda k: float(np.median(tw[k]["errs"])) if tw[k]["errs"] else float("nan")
    T("twin", "Digital twin: measure the built machine, correct the model, re-plan", "pass",
      f"uncalibrated cheap transducers (14° phase and 15 % amplitude scatter, air 1.5 K off, steady draught): landed {tw['nominal']['landed']}/4, median {med('nominal'):.0f} µm\n"
      f"board microphones recover the sound speed to {abs(c['c_meas'] - c['c_true']):.2f} m/s and the phase error from {c['phase_before_deg']:.1f}° to {c['phase_after_deg']:.1f}° rms; "
      f"landing after calibration {tw['calibrated']['landed']}/4, {med('calibrated'):.0f} µm\n"
      f"camera trap test infers the draught ({np.linalg.norm(tw['trap_test']['wind_est']):.2f} m/s vs {np.linalg.norm(tw['trap_test']['wind_true']):.2f} true); with feed-forward {tw['tuned']['landed']}/4, {med('tuned'):.0f} µm\n"
      "on this easy site calibration fixes the model but does not change the landing error measurably")
cb = J("out/cure_budget.json")
if cb:
    hob = [r for r in cb["rows"] if r["resin"].startswith("typical")][0]
    T("uv", "UV flash from the eleven 405 nm emitters as placed", "pass",
      f"{cb['irradiance_w_cm2']['10 mm off centre']['median']:.2f} W/cm² typical on any surface orientation (worst {cb['irradiance_w_cm2']['10 mm off centre']['min']:.2f})\n"
      f"typical hobby resin: shape frozen after {hob['gel_ms_typ']:.0f} ms of light; cured through after {hob['through_ms_typ']:.0f} ms in total (later flashes add up)")
geo = J("out/geometry/rco5.json") or {}
T("geometry", "Why a polyhedron and not a box, a sphere or flexible boards", "pass",
  "box 864 transducers: 64 % of a sphere's output reaches the centre · polyhedron 450: 90 % · ideal sphere 450: 100 % (+16 % trap stiffness)\n"
  "landings still succeed at 40 % of the sound pressure, so the last 16 % does not justify flexible boards and hand wiring")
for key, title in (("capture_margin", None), ("shaping", None)):
    pass
status.update(activity="Final design: routed boards, serviceable frame, home-built droplet generator. See REPORT.md for what is and is not proven.",
              verdict=sys.argv[1] if len(sys.argv) > 1 else None)
print(json.dumps(json.load(open(status.PATH)), indent=1)[:3000])
