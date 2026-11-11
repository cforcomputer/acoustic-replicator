"""Fill the generated tables of REPORT.md (bill of materials, print results) from the result files.
The report keeps the markers, so this can be re-run after any test.  Usage: s19_report_tables.py"""
import json, os, re
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
J = lambda p: json.load(open(f"{ROOT}/{p}")) if os.path.exists(f"{ROOT}/{p}") else None


def block(text, name, body):
    a, b = f"<!-- {name} -->", f"<!-- /{name} -->"
    if a in text:
        return re.sub(re.escape(a) + r".*?" + re.escape(b), lambda m: a + "\n" + body + "\n" + b, text, flags=re.S)
    return text.replace(name, a + "\n" + body + "\n" + b, 1)


rep = open(f"{ROOT}/REPORT.md").read()

bom = J("hardware/bom.json")
rows = ["| Item | Qty | Unit (USD) | Line (USD) | Supplier | Status |", "|---|---|---|---|---|---|"]
for b in bom["bom"]:
    rows.append(f"| {b['item']} | {b['qty']} | {b['unit_usd']:.2f} | {b['total_usd']:.0f} | {b['where']} | {b['status']} |")
n_ver = sum(1 for b in bom["bom"] if b["status"] == "verified")
rows.append(f"| **Total** | | | **{bom['total_usd']:.0f}** | | {n_ver} of {len(bom['bom'])} lines verified |")
rep = block(rep, "BOM_TABLE", "\n".join(rows))
rep = re.sub(r"\d+ of \d+ lines were opened on a supplier page today", f"{n_ver} of {len(bom['bom'])} lines were opened on a supplier page today", rep)

# printed parts and fasteners, from the hardware build
pp = J("hardware/print/printed_parts.json")
what = dict(frame_cap="Frame cap: top port and the six upper board plates; lifts off", frame_mid="Middle frame: upper halves of the six equatorial plates, three camera ports",
            frame_lower="Lower frame: nine board plates, bottom port, three strobe ports, three leg bosses", chuck="Build-pin chuck", dock="Cartridge dock",
            jet_body="Droplet-generator body with drip tray", jet_clamp="Droplet-generator clamp ring", gate_bracket="Light-gate bracket", res_clamp="Reservoir clamp")
if pp:
    rows = ["| Part | Size (mm) | Volume (cm3) | One solid | Survives the weak-joint test |", "|---|---|---|---|---|"]
    for k, v in pp.items():
        nm = what.get(k, "Leg" if k.startswith("stand_leg") else "Camera pod" + (" with sensor shelf" if v["volume_cm3"] > 7 else ""))
        rows.append(f"| {nm} (`{k}.stl`) | {v['size_mm'][0]:.0f} x {v['size_mm'][1]:.0f} x {v['size_mm'][2]:.0f} | {v['volume_cm3']:.1f} | {'yes' if v['solids'] == 1 else 'NO'} | {'yes' if v['neck']['pieces_after_peeling'] == 1 else 'NO'} |")
    rows.append(f"| **Total: {len(pp)} parts** | | **{sum(v['volume_cm3'] for v in pp.values()):.0f}** | | |")
    rep = block(rep, "PRINTED_TABLE", "\n".join(rows))
fz = J("hardware/fasteners.json")
if fz:
    rows = ["| Fastener | Count in the model |", "|---|---|"] + [f"| {k} | {v} |" for k, v in fz.items()]
    rows.append(f"| **Total** | **{sum(fz.values())}** |")
    rep = block(rep, "FASTENER_TABLE", "\n".join(rows))

names = [("", "One droplet size (0.3 mm); landings scatter as on the smooth test part"),
         ("_fine", "0.3 mm core + 0.2 mm outer shell (0.2 mm deep); landings scatter as on the smooth test part"),
         ("_mc", "One droplet size (0.3 mm); landing errors and losses drawn from the flights onto Benchy geometry")]
rows = ["| Print | Droplets | Machine time | Surface error, mean (µm) | on faces | on edges | Surface off by > 0.1 mm | Internal voids | Sites built |",
        "|---|---|---|---|---|---|---|---|---|"]
T_DOWN, T_OTHER = 0.159, 0.37            # s per droplet incl. the 20 ms flash: direct ascent (lands at 0.135-0.148 s) / carry (0.33-0.38 s), from the landing tests
for tag, name in names:
    r, e = J(f"out/print{tag}/result.json"), J(f"out/print{tag}/edges.json")
    if not (r and e):
        rows.append(f"| {name} | still computing | | | | | | | |")
        continue
    nd = r["normals"]
    t_h = (nd["down"] * T_DOWN + (nd["side"] + nd["up"]) * T_OTHER) / 3600
    t_h *= 1 + r["stats"]["lost"] / max(r["droplets"], 1)      # a lost droplet costs a cycle too
    off = 100 * (e["all"]["missing_100um"] + e["all"]["proud_100um"])
    rows.append(f"| {name} | {r['droplets']:,} | about {t_h:.0f} h | {e['all']['mean_abs_um']:.0f} | {e['faces']['mean_abs_um']:.0f} | {e['edges']['mean_abs_um']:.0f} | "
                f"{off:.0f} % | {100*r['interior_void_fraction']:.1f} % | {100*r['completeness']:.1f} % |")
    if tag == "_fine":
        rep = re.sub(r"takes about (PRINT_FINE_TIME|<!-- FT -->.*?<!-- /FT -->)", f"takes about <!-- FT -->{t_h:.0f} h<!-- /FT -->", rep, flags=re.S)
rep = block(rep, "PRINT_TABLE", "\n".join(rows))
open(f"{ROOT}/REPORT.md", "w").write(rep)
print("\n".join(rows))
