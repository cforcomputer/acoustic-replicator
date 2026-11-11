"""Build the final hardware model and check it.

Writes: hardware/replicator_mk2.step, hardware/print/*.stl (printed parts), hardware/machine.json + meshes.bin (viewer),
hardware/bom.json, hardware/interference_report.json, hardware/attachment_report.json, hardware/wiring.json.
Usage: s16_hardware.py [--fast]   (--fast skips the STEP / STL export)
"""
import sys, json, time
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator.cad import Machine, build_solids, interference_report, attachment_report, wire_report, motion_report, neck_report, export, _solid, WIRE, MM

m = Machine()
t0 = time.time()
asm = m.build()
parts = asm["parts"]
print(f"{len(parts)} parts, {len(asm['wires'])} wire conductors ({time.time()-t0:.1f}s)")
solids = [_solid(c) for c in parts]
print(f"solids built ({time.time()-t0:.0f}s)")
bad = [p["id"] for p, s in zip(parts, solids) if not (hasattr(s, "volume") and s.volume > 1e-6)]
print("degenerate solids:", bad[:10])

checked, clashes = interference_report(parts, solids)
print(f"interference: {checked} close pairs checked, {len(clashes)} overlaps ({time.time()-t0:.0f}s)")
for c in sorted(clashes, key=lambda x: -x[2])[:60]:
    print(f"   {c[0]:28s} x {c[1]:28s} {c[2]:9.2f} mm3")
att = attachment_report(parts, solids)
print(f"attachment: {att['n_parts']} parts; without a fastening: {len(att['unmounted'])}; not touching what they are fastened to: {len(att['gaps'])}; "
      f"not connected to the floor: {len(att['unrooted'])}; printed bodies in more than one piece: {att['split_bodies']} ({time.time()-t0:.0f}s)")
for g in att["unmounted"][:10]: print("   no mount:", g)
for g in att["gaps"][:25]: print("   gap:", g)
print("   printed bodies (pieces):", att["bodies"])
wr = wire_report(parts, solids, asm["wires"])
print(f"wiring: {wr['n_wires']} conductors, {wr['length_m']} m; loose ends: {len(wr['dangling'])}; runs through a solid: {len(wr['through'])} ({time.time()-t0:.0f}s)")
for d in wr["dangling"][:12]: print("   loose:", d)
for d in sorted(wr["through"], key=lambda x: -x[2])[:25]: print("   through:", d)

# ---- things that move: do they clear everything on their way?
import build123d as bd
E = m._enc_dims()
b_, sh_, zt_, f_ = (E[k] / MM for k in ("b", "sh", "zt", "f"))
ix = lambda pred: [i for i, p in enumerate(parts) if pred(p)]
cap_tags = [bd_["name"] for bd_ in asm["boards"] if bd_.get("cap")]
on_cap = lambda p: p.get("body") == "frame_cap" or any(p["id"].startswith(t + ".") for t in cap_tags) or (p["group"] == "transducers" and p["mount"]["to"].split(".")[0] in cap_tags)
chuck = lambda p: p.get("body") == "chuck" or p["id"] == "pin" or p["id"].startswith("chuck.")
lid = lambda p: p["group"] == "lid" or p["id"].startswith("hinge.lid")
door = lambda p: p["id"] in ("enc.door", "foam.door", "door.window") or p["id"].startswith("door.window.s")
jet = lambda p: p.get("body") in ("jet_body", "jet_clamp") or (p["id"].startswith("jet.") and not p["id"].startswith("jet.lock"))
part_env = bd.Cylinder(15.0, 30.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)).moved(bd.Location((0, 0, m.pole_tip_z / MM)))
bq = next(bd_ for bd_ in asm["boards"] if not bd_.get("cap") and abs(bd_["c"][2]) < 1e-6)
nq = np.array(bq["n"])
motions = [
    dict(name="lift the build-pin chuck, pin and a 30 mm part out through the top port (lid open)", moving=ix(lambda p: chuck(p) and not p["id"].startswith("chuck.hold")),
         ignore=ix(lambda p: lid(p) or p["id"].startswith("chuck.hold")), steps=[("move", (0, 0, dz)) for dz in (3, 10, 30, 60, 100, 135)], extra=[("printed part, 30 mm", part_env)]),
    dict(name="lift the frame cap with its six boards straight off (chuck removed, lid open)", moving=ix(on_cap), ignore=ix(lambda p: chuck(p) or lid(p)),
         steps=[("move", (0, 0, dz)) for dz in (2, 8, 20, 50, 100)]),
    dict(name="slide the droplet-generator cartridge out to the front (lock screw removed, door open)", moving=ix(jet),
         ignore=ix(lambda p: p["id"].startswith("jet.lock") or door(p) or p["id"].startswith("hinge.door")), steps=[("move", (0, -dy, 0)) for dy in (3, 10, 20, 40, 70, 110)]),
    dict(name="swing the door open", moving=ix(door), ignore=ix(lambda p: p["id"].startswith("hinge.door")),
         steps=[("turn", (-(b_ + sh_), -(b_ + sh_), 0), (0, 0, 1), -ang) for ang in (10, 30, 60, 95)]),
    dict(name="swing the lid open", moving=ix(lambda p: p["group"] == "lid"), ignore=ix(lambda p: p["id"].startswith("hinge.lid")),
         steps=[("turn", (0, b_ + sh_, zt_ + f_), (1, 0, 0), -ang) for ang in (10, 30, 60, 95)]),
    dict(name=f"pull one board ({bq['name']}) straight out of its pocket (4 screws and 2 plugs removed)",
         moving=ix(lambda p: p["id"] in (bq["name"] + ".pcb", bq["name"] + ".parts") or (p["group"] == "transducers" and p["mount"]["to"] == bq["name"] + ".pcb")),
         ignore=ix(lambda p: p["id"].startswith(bq["name"] + ".screw")), steps=[("move", tuple(nq * dd)) for dd in (2, 8, 25)]),
]
mot = motion_report(parts, solids, motions)
for r in mot:
    print(f"motion: {r['name']}: {r['moving_parts']} parts, {r['steps']} positions, {r['pairs_checked']} pairs checked, collisions: {len(r['collisions'])}")
    for c in r["collisions"][:6]:
        print("      ", c)
json.dump(mot, open(f"{ROOT}/hardware/motion_report.json", "w"), indent=1)
print(f"   ({time.time()-t0:.0f}s)")

# self-test of the checks: inject one clash, one floating part, one wire through a board
if "--selftest" in sys.argv:
    import copy
    from replicator.cad import box, cyl, mount
    p2 = parts + [box("test.clash", "test", "injected clash", [0.0, 0.0, 0.085], [0.01, 0.01, 0.01], "#f00", mount=mount("enc.floor", "none")),
                  box("test.float", "test", "injected floating part", [0.05, 0.05, 0.12], [0.01, 0.01, 0.01], "#f00")]
    s2 = solids + [_solid(p2[-2]), _solid(p2[-1])]
    _, cl = interference_report(p2, s2)
    at = attachment_report(p2, s2)
    w2 = [dict(label="test wire", fn="gnd", color="#000", r=0.0005, a="pin", b="jet.block", pts=[[0, 0, 0.11], [0, 0, 0.0], [0.08, 0, 0.0], [0.12, 0, 0.0]])]
    mt = motion_report(p2[:len(parts)] + [p2[-2]], solids + [s2[-2]], [dict(name="test", moving=ix(chuck), ignore=ix(lid), steps=[("move", (0, 0, 3))])])
    wt = wire_report(p2, s2, w2)
    print("self-test: clash found:", any("test.clash" in (c[0], c[1]) for c in cl), "| floating part found:", "test.float" in at["unmounted"],
          "| gap found:", any(g[0] == "test.clash" for g in at["gaps"]), "| wire through solid found:", len(wt["through"]) > 0, "| loose end found:", len(wt["dangling"]) > 0, "| collision in a motion found:", len(mt[0]["collisions"]) > 0)

json.dump(dict(pairs_checked=checked, overlaps=[dict(a=a, b=b, mm3=v) for a, b, v in clashes], n_solids=len(solids)), open(f"{ROOT}/hardware/interference_report.json", "w"), indent=1)
json.dump(att, open(f"{ROOT}/hardware/attachment_report.json", "w"), indent=1)
by_fn = {}
for w in asm["wires"]:
    L = float(np.linalg.norm(np.diff(np.array(w["pts"]), axis=0), axis=1).sum())
    by_fn.setdefault(w["fn"], [0, 0.0]); by_fn[w["fn"]][0] += 1; by_fn[w["fn"]][1] += L
cables = {}
for w in asm["wires"]:
    cables.setdefault(w["label"], dict(from_part=w["a"], to_part=w["b"], conductors=[], length_mm=0))
    cables[w["label"]]["conductors"].append(w["fn"])
    cables[w["label"]]["length_mm"] = max(cables[w["label"]]["length_mm"], round(1e3 * float(np.linalg.norm(np.diff(np.array(w["pts"]), axis=0), axis=1).sum())))
json.dump(dict(report=wr, legend={k: dict(color=c, meaning=mm) for k, (c, mm) in WIRE.items()},
               by_function={k: dict(conductors=v[0], length_m=round(v[1], 2)) for k, v in by_fn.items()}, cables=cables), open(f"{ROOT}/hardware/wiring.json", "w"), indent=1)
data = m.to_json(f"{ROOT}/hardware/machine.json", f"{ROOT}/hardware/meshes.bin", asm)
json.dump(dict(bom=data["bom"], total_usd=data["bom_total_usd"]), open(f"{ROOT}/hardware/bom.json", "w"), indent=1)
json.dump(m.fasteners(), open(f"{ROOT}/hardware/fasteners.json", "w"), indent=1)
print(f"machine.json written; BOM total ${data['bom_total_usd']:.0f} ({time.time()-t0:.0f}s)")
lo = np.array([[b.min.X, b.min.Y, b.min.Z] for b in (s.bounding_box() for s in solids)]).min(0)
hi = np.array([[b.max.X, b.max.Y, b.max.Z] for b in (s.bounding_box() for s in solids)]).max(0)
print("overall size (mm):", np.round(hi - lo, 0))
if "--fast" not in sys.argv:
    pr = export(m, f"{ROOT}/hardware", parts, solids)
    for name in pr:                                              # does each printed part hold together by more than a thread?
        pr[name]["neck"] = neck_report(f"{ROOT}/hardware/print/{name}.stl")
    json.dump(pr, open(f"{ROOT}/hardware/print/printed_parts.json", "w"), indent=1)
    print("neck check (pieces left after peeling 0.8 mm off every surface; 1 = sound):", {k: v["neck"]["pieces_after_peeling"] for k, v in pr.items()})
    print("printed bodies:", json.dumps(pr))
    print(f"STEP + STL written ({time.time()-t0:.0f}s)")
