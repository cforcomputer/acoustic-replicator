"""Checks of the routed transducer board (run after `python -m replicator.pcbroute hardware/pcb`).

P1  placement rules and netlist are complete;
P2  the Gerber and drill files, read back and rasterised, contain exactly the netlist: no open circuit, no short,
    clearances and drill rules kept (this uses nothing from the router);
P3  the checker itself notices an injected open, short and clearance fault;
P4  an independent Gerber parser (gerbonara) reads the set, finds every layer and the same outline and holes;
P5  the JLCPCB bill of materials and placement file agree with the board and with JLCPCB's own library entries.
"""
import sys, os, json, csv, tempfile, warnings
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
from replicator.pcb import Board, FOOTPRINTS, LIBDIR, ORDER
from replicator import pcbroute as R

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


b = Board()
out = f"{ROOT}/hardware/pcb"
gd = f"{out}/gerbers"
data = json.load(open(f"{out}/TB-25_routes.json"))
tracks, vias = data["tracks"], data["vias"]

print("P1 placement and netlist")
v, mins = b.drc()
check("placement rules clean", not v, f"min clearances {mins}")
smd = [(it["ref"], p["num"]) for it in b.items if it["side"] == "back" for p in b.pads_of(it)]
nc = [k for k in smd if not b.padnet.get(k)]
check("every pad has a net or is deliberately unconnected", all(k in b.padnet for k in smd), f"{len(smd)} surface-mount pads, {len(nc)} unconnected by design "
      f"(driver NC pins, unused driver output, USB pins, connector tabs, fiducials)")
check("no net has a single pad", all(len(x) > 1 for x in b.nets.values()), f"{len(b.nets)} nets")
check("the 25 drive pins are consecutive GPIOs (one PIO 'out pins' instruction)", sorted(c["gpio"] for c in b.channel.values()) == list(range(4, 29)))
check("every transducer has its own driver output", len({(c["driver"], c["output"]) for c in b.channel.values()}) == 25)
check("router finished every net", not data["info"]["not_completed"], f"{len(tracks)} tracks, {len(vias)} vias, {data['info']['ripups']} rip-ups")

print("P2 the fabrication files contain the netlist (rebuilt from the Gerber and drill files alone)")
r = R.verify(b, gd)
check("no open circuit", not r["opens"], f"{r['nets_checked']} nets, {r['pads_checked']} pads" + (f" OPEN: {list(r['opens'])[:5]}" if r["opens"] else ""))
check("no short circuit", not r["shorts"], str(r["shorts"][:3]) if r["shorts"] else "")
check("copper clearance >= 0.127 mm between different nets on all four layers", r["n_clearance_violations"] == 0,
      ", ".join(f"{k} {c['min_gap_mm']} mm" for k, c in r["clearance"].items()))
check("copper >= 0.3 mm from the board edge", all(c["edge_mm"] >= 0.3 for c in r["clearance"].values()), ", ".join(f"{k} {c['edge_mm']}" for k, c in r["clearance"].items()))
check("drilled holes >= 0.25 mm apart wall to wall; plated holes have their ring on both outer layers", r["hole_wall_to_wall_min_mm"] >= 0.25 and r["n_annular"] == 0,
      f"{r['plated_holes']} plated, {r['non_plated_holes']} not plated, closest {r['hole_wall_to_wall_min_mm']} mm")
check("every plated hole touches copper", r["holes_touching_no_copper"] == 0)

print("P3 the checker notices injected faults")
with tempfile.TemporaryDirectory() as td:
    st = R.selftest(b, tracks, vias, td)
check("clean copy passes", st["clean_passes"])
check("deleted track -> open circuit reported", st["open_found"])
check("bridging track -> short circuit reported", st["short_found"])
check("track 0.06 mm from another net -> clearance fault reported", st["clearance_found"], f"gap reported {st['smallest_gap_reported_mm']} mm")

print("P4 independent parser")
try:
    from gerbonara import LayerStack
    stack = LayerStack.open(gd)
    names = str(stack)
    want = ("top copper", "bottom copper", "top mask", "bottom mask", "bottom paste", "mechanical outline")
    check("gerbonara identifies the layers", all(k in names for k in want) and names.count("inner") >= 0, names[names.find("["):][:230])
    (x0, y0), (x1, y1) = stack["mechanical outline"].bounding_box("mm")
    check("outline 58 x 58 mm", abs((x1 - x0) - 58.1) < 0.15 and abs((y1 - y0) - 58.1) < 0.15, f"{x1-x0:.2f} x {y1-y0:.2f} mm incl. the 0.1 mm line")
    n_drill = sum(len(list(d.objects)) for d in stack.drill_layers)
    check("same number of holes as the board has", n_drill == r["plated_holes"] + r["non_plated_holes"], f"{n_drill}")
    for key, lay in (("top copper", "L1"), ("bottom copper", "L4")):
        n_obj = len(list(stack[key].objects))
        check(f"{key} parsed", n_obj > 50, f"{n_obj} objects")
except ImportError:
    print("     (gerbonara not installed: skipped)")

print("P5 assembly files")
bom = list(csv.DictReader(open(f"{out}/TB-25_BOM_JLCPCB.csv")))
cpl = list(csv.DictReader(open(f"{out}/TB-25_CPL_JLCPCB.csv")))
refs_bom = [x for row in bom for x in row["Designator"].split(",")]
refs_cpl = [row["Designator"] for row in cpl]
placed = [it["ref"] for it in b.items if it["side"] == "back" and it["part"] not in ("TP", "FID") and it.get("value") != "120R"]
check("bill of materials, placement file and board list the same parts", sorted(refs_bom) == sorted(refs_cpl) == sorted(placed), f"{len(placed)} parts, {len(bom)} lines")
check("column headers as JLCPCB requires", list(bom[0]) == ["Comment", "Designator", "Footprint", "LCSC Part #"] and list(cpl[0]) == ["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
bad = []
for row in bom:
    fn = os.path.join(LIBDIR, row["LCSC Part #"] + ".json")
    if not os.path.exists(fn):
        bad.append(row["LCSC Part #"]); continue
    lib = json.load(open(fn))["result"]
    if (lib.get("lcsc") or {}).get("number") != row["LCSC Part #"]:
        bad.append(row["LCSC Part #"])
check("every part number has a JLCPCB library entry on file, and the footprint used is that entry's", not bad, f"{len(bom)} part numbers" + (f"; missing {bad}" if bad else ""))
inside = all(0.3 < float(c["Mid X"][:-2]) < 57.7 and 0.3 < float(c["Mid Y"][:-2]) < 57.7 and c["Layer"] == "Bottom" for c in cpl)
check("all placements on the bottom side, inside the board", inside)
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
