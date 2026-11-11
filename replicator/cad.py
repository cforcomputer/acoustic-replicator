"""Final-assembly hardware model (Replicator Mk2), built on real part dimensions.

One parametric description (`Machine`) drives four things so they cannot drift
apart: the acoustic simulation (transducer positions come from `configs.rco`),
the transducer board (`pcb.Board`), the STEP / STL files for building it, and the
scene used by the viewer.

What is modelled
  * the printed holding structure: a two-piece frame (one plate per face of the
    polyhedron, with a window, a locating pocket and screw holes for each board),
    the stand with the drip cup, jet mount and legs, three camera pods, and the
    lift-out top plate that carries the build pin;
  * every bought module at its real outline size, and every wire.

What is checked (functions at the bottom)
  * interference_report : no two solids overlap;
  * attachment_report   : every part names what it is fastened to and how, really
                          touches it, and the chain of fastenings reaches the floor
                          of the enclosure - nothing floats;
  * wire_report         : every wire starts and ends on a part, and does not pass
                          through a solid.

Coordinates: origin = centre of the build volume, +z up, metres.
"""
from dataclasses import dataclass, asdict
import itertools
import json
import os
import numpy as np

from . import configs
from .delivery import JET_MK2
from .pcb import Board

MM = 1e-3

# colours ---------------------------------------------------------------------
COL = dict(frame="#23262b", stand="#2d3138", pod="#2d3138", top="#e07b1a", pcb="#1e6b3c", pcb_black="#15171a", can="#c9ccd1",
           steel="#b9bec6", brass="#c8a04a", alu="#cfd3d8", foam="#d9d4c7", ply="#b98d5a", acrylic="#f08a1c", lens="#101216",
           glass="#9fd2ff", uv="#8a3ffc", red="#e23b3b", white="#f1f1ee", black="#1b1b1f", resin="#e9d9a8", fan="#17181b")
# what every wire colour means (shown as a legend in the viewer)
WIRE = {
    "drive+":   ("#d11f1f", "transducer drive supply (+18 to +24 V)"),
    "gnd":      ("#101114", "ground / return"),
    "5v":       ("#f08a1c", "+5 V logic supply"),
    "rs485a":   ("#f2d21b", "command bus RS-485 A"),
    "rs485b":   ("#1faa4b", "command bus RS-485 B"),
    "sync":     ("#f2f2f2", "SYNC: restarts every board's 40 kHz period together"),
    "uv+":      ("#8a3ffc", "405 nm LED string (+)"),
    "strobe+":  ("#8a5a2b", "red strobe LED (+)"),
    "usb":      ("#3a3d44", "camera USB cable"),
    "trigger":  ("#58b7f0", "camera trigger"),
    "valve":    ("#1f5fd1", "piezo pulse of the droplet generator (+/- up to 30 V)"),
    "heater":   ("#e11d48", "nozzle heater supply"),
    "therm":    ("#84cc16", "nozzle thermistor"),
    "gate":     ("#9aa0a6", "droplet light gate signal"),
    "laser":    ("#f06fb0", "line laser supply"),
    "3v3":      ("#ff6b6b", "+3.3 V sensor supply"),
    "sda":      ("#3b82f6", "I2C data"),
    "scl":      ("#eab308", "I2C clock"),
    "interlock": ("#d8d8d8", "door / lid interlock loop"),
    "tec+":     ("#b91c1c", "Peltier supply (12 V, through the H-bridge)"),
    "fan":      ("#6b7280", "fan supply / PWM"),
    "resin":    ("#e9e2c8", "resin feed tube (PTFE)"),
}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def perp(v):
    """Any unit vector perpendicular to v."""
    v = unit(v)
    a = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return unit(np.cross(v, a))


def _L(v):
    return [float(x) for x in v]


def box(pid, group, label, center, size, color, x=(1, 0, 0), z=(0, 0, 1), holes=None, cuts=None, **kw):
    return dict(id=pid, group=group, kind="box", label=label, center=_L(center), size=_L(size), x=_L(x), z=_L(z),
                color=color, holes=holes or [], cuts=cuts or [], **kw)


def cyl(pid, group, label, top, axis, r, h, color, bore=0.0, holes=None, **kw):
    """Cylinder whose end face `top` is the +axis end; the body extends h back along -axis.  bore > 0 makes it a tube."""
    return dict(id=pid, group=group, kind="cyl", label=label, top=_L(top), axis=_L(unit(axis)), r=float(r), h=float(h),
                color=color, bore=float(bore), holes=holes or [], **kw)


def drill(p, axis, r):
    return dict(p=_L(p), axis=_L(unit(axis)), r=float(r))


def mount(to, by):
    return dict(to=to, by=by)


@dataclass
class Machine:
    """Final design: 18 identical 5x5 transducer boards on the square faces of a
    rhombicuboctahedron; a triangular port straight up (build pin) and straight down
    (jet + drip cup); cameras, strobes and UV emitters in the six side ports."""
    name: str = "Replicator Mk2"
    n: int = 5
    vpp: float = 18.0
    can_d: float = 9.9 * MM
    can_h: float = 7.1 * MM
    pcb_t: float = 1.6 * MM
    back_h: float = 5.2 * MM          # tallest part on the back of a board (power connector)
    lip: float = 1.0 * MM             # depth of the locating pocket each board sits in
    port_d: float = 31.0 * MM         # clear opening of the top and bottom ports
    pole_d: float = 2.5 * MM
    pole_tip_z: float = 10.0 * MM
    pole_len: float = 100.0 * MM
    nozzle_tip_z: float = JET_MK2["z_nozzle"]
    acquire_z: float = JET_MK2["z_acq"]
    park_z: float = JET_MK2["z_park"]
    jet_tilt_deg: float = 1.0         # a missed droplet falls back ~5 mm beside the nozzle, into the cup
    enc_clear_xy: float = 135.0 * MM
    enc_clear_top: float = 150.0 * MM
    enc_clear_bot: float = 142.0 * MM
    foam_t: float = 25.4 * MM
    shell_t: float = 6.35 * MM
    drawer_h: float = 60.0 * MM
    temp_setpoint_c: float = 28.0
    temp_band_k: float = 0.2
    build_d: float = 30.0 * MM
    build_h: float = 30.0 * MM
    cam_f_mm: float = 2.8             # lens supplied with the camera
    cam_sensor_mm: tuple = (3.84, 2.40)
    cam_px: tuple = (1280, 800)

    # ------------------------------------------------------------------ geometry
    def array(self):
        return configs.rco(self.n, vpp=self.vpp)

    @property
    def half(self):
        return self.array().r_face

    def geometry(self):
        """Faces of the frame.  Squares carry boards, triangles are ports."""
        arr = self.array()
        a, rs, rt = arr.edge, arr.r_face, arr.r_port
        fo = (rs + self.can_h + self.lip) / rs                    # outer surface of the frame = inner surface scaled by fo
        sq = []
        for i, pan in enumerate(arr.panels):
            n = -pan.normal
            corners = [pan.center + sx * a / 2 * pan.u + sy * a / 2 * pan.v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            sq.append(dict(i=i, name=pan.name, n=n, u=pan.u, v=pan.v, c=pan.center, verts=np.array(corners)))
        tri = []
        g = 1 + np.sqrt(2)
        for s in itertools.product((1, -1), repeat=3):
            s = np.array(s, float)
            d = configs.ROT3 @ (s / np.sqrt(3))
            verts = np.array([configs.ROT3 @ (a / 2 * s * np.where(np.arange(3) == k, g, 1.0)) for k in range(3)])
            c = d * rt
            role = "top" if d[2] > 0.9 else "bottom" if d[2] < -0.9 else "upper" if d[2] > 0 else "lower"
            # vertex 0 = the odd one out in height (the tip that crosses the equator on side ports)
            zz = verts[:, 2]
            k0 = int(np.argmin(zz)) if role in ("upper", "top") else int(np.argmax(zz))
            order = [k0, (k0 + 1) % 3, (k0 + 2) % 3]
            verts = verts[order]
            if np.dot(np.cross(verts[1] - verts[0], verts[2] - verts[0]), d) < 0:      # counter-clockwise seen from outside
                verts = verts[[0, 2, 1]]
            w = np.array([unit(v - c) for v in verts])
            tri.append(dict(role=role, d=d, c=c, verts=verts, w=w, az=float(np.arctan2(d[1], d[0]))))
        key = lambda t: t["az"]
        ports = dict(top=[t for t in tri if t["role"] == "top"][0], bottom=[t for t in tri if t["role"] == "bottom"][0],
                     upper=sorted([t for t in tri if t["role"] == "upper"], key=key),
                     lower=sorted([t for t in tri if t["role"] == "lower"], key=key))
        return dict(arr=arr, a=a, rs=rs, rt=rt, fo=fo, sq=sq, ports=ports, t_tri=rt * (fo - 1))

    def ports(self):
        g = self.geometry()["ports"]
        return dict(top=g["top"]["d"], bottom=g["bottom"]["d"], upper=[t["d"] for t in g["upper"]], lower=[t["d"] for t in g["lower"]])

    # ------------------------------------------------------------------ fasteners
    SCREWS = {"M2": (1.0, 1.9, 2.0), "M2.5": (1.25, 2.25, 2.5), "M3": (1.5, 2.75, 3.0), "M4": (2.0, 3.5, 4.0), "M5": (2.5, 4.25, 5.0), "3.5 mm": (1.75, 3.5, 2.4)}   # shank r, head r, head h [mm]
    INSERTS = {"M2.5": (1.9, 4.0), "M3": (2.3, 5.0), "M4": (3.15, 7.0), "M5": (3.6, 8.0)}                                                                            # outer r, length [mm]

    def _screw(self, P, pid, size, L, p, axis, through=(), into=None, grip=0.0, insert=False, thumb=False, wood=False, by=None, group="fasteners"):
        """Add a screw: head seated at p, pointing along `axis` (head to tip), length L [mm].  `through` = ids or printed-body
        names it passes with clearance, `into` = what it threads into, `grip` = clamped thickness [mm];
        insert=True adds a brass heat-set insert where the thread starts.  Holes are cut afterwards by _cut_fastener_holes."""
        rs, rh, hh = self.SCREWS[size]
        a = unit(axis)
        p = np.asarray(p, float)
        if thumb:
            rh, hh = 5.0, 7.0
        name = f"{size} x {L:g} mm " + ("knurled thumb screw" if thumb else "wood screw, pan head" if wood else "socket-head screw")
        tgt = into or (through[-1] if through else None)
        P.append(cyl(pid, group, name, p + a * L * MM, a, rs * MM, L * MM, COL["steel"], mat="metal",
                     mount=mount(pid + ".insert" if insert else tgt, by or ("threaded into its heat-set insert" if insert else "threaded into the part"))))
        P.append(cyl(pid + ".head", group, name + " (head)", p, a, rh * MM, hh * MM, COL["black"] if thumb else COL["steel"], mat="plastic" if thumb else "metal",
                     mount=mount(pid, "one piece with its screw")))
        if insert:
            ri, li = self.INSERTS[size]
            P.append(cyl(pid + ".insert", group, f"{size} heat-set insert (brass)", p + a * grip * MM, -a, ri * MM, li * MM, COL["brass"], bore=rs * MM, mat="metal",
                         mount=mount(into, "melted into the printed part with a soldering iron")))
        self._fasteners.append(dict(id=pid, size=size, L=L, p=p, a=a, through=list(through), into=into, grip=grip, insert=insert, thumb=thumb, name=name))

    def _cut_fastener_holes(self, P):
        """Clearance holes in everything a screw passes, a tapping hole where it threads, a pocket for its insert."""
        def extent(c):
            if c["kind"] == "box":
                return np.array(c["center"]), 0.5 * float(np.linalg.norm(c["size"]))
            if c["kind"] == "cyl":
                return np.array(c["top"]) - np.array(c["axis"]) * c["h"] / 2, float(np.hypot(c["r"], c["h"] / 2))
            pts = np.array(c["v_in"] + c["v_out"]) if c["shape"] == "frustum" else np.vstack([np.array(c["base"]), np.array(c["base"]) + np.array(c["vec"])])
            cen = pts.mean(0)
            return cen, float(np.linalg.norm(pts - cen, axis=1).max())
        byid = {c["id"]: c for c in P}
        bodies = {}
        for c in P:
            if c.get("body"):
                bodies.setdefault(c["body"], []).append(c)
        for p0, ax, r, h in self._reliefs:                  # notches in the cap, open downwards, so that it lifts straight off
            for c in bodies.get("frame_cap", []):
                cen, rad = extent(c)
                if np.linalg.norm(cen - (p0 + ax * h / 2)) > rad + h:
                    continue
                c["ops"] = list(c.get("ops", [])) + [dict(op="cyl", p=_L(p0 + np.array([0, 0, dz * MM])), axis=_L(ax), r=r, h=h) for dz in range(0, -19, -2)]
        for f in self._fasteners:
            if f.get("board"):
                continue
            rs = self.SCREWS[f["size"]][0]
            cuts = [(t, rs + 0.2, -0.3, f["L"] + 0.3) for t in f["through"]]
            if f["into"]:
                cuts.append((f["into"], rs + 0.05, f["grip"] - 0.2, f["L"] + 0.5))
                if f["insert"]:
                    ri, li = self.INSERTS[f["size"]]
                    cuts.append((f["into"], ri, f["grip"] - 0.05, f["grip"] + li + 0.05))
            for tgt, r_hole, s0, s1 in cuts:
                for c in ([byid[tgt]] if tgt in byid else bodies.get(tgt, [])):
                    cen, rad = extent(c)
                    mid = f["p"] + f["a"] * 0.5 * (s0 + s1) * MM
                    if np.linalg.norm(cen - mid) > rad + 0.5 * (s1 - s0) * MM + 2 * MM:
                        continue
                    c["ops"] = list(c.get("ops", [])) + [dict(op="cyl", p=_L(f["p"] + f["a"] * s0 * MM), axis=_L(f["a"]), r=r_hole * MM, h=(s1 - s0) * MM)]
                    if any(o.get("op") == "half" for o in c["ops"]):            # the split must stay the last operation
                        c["ops"] = [o for o in c["ops"] if o.get("op") != "half"] + [o for o in c["ops"] if o.get("op") == "half"]

    def fasteners(self):
        """Counts by type, from the screws actually drawn in the model."""
        out = {}
        for f in self._fasteners:
            out[f["name"]] = out.get(f["name"], 0) + 1
            if f.get("insert"):
                k = f"{f['size']} heat-set insert (brass)"
                out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    # ------------------------------------------------------------------ assembly
    def build(self):
        """Returns dict(parts=[...], wires=[...], cameras=[...], lights=[...], boards=[...])."""
        G = self.geometry()
        arr, a, rs, rt, fo, t_tri = G["arr"], G["a"], G["rs"], G["rt"], G["fo"], G["t_tri"]
        brd = Board(self.n)
        side = brd.side * MM
        hm = (brd.side / 2 - 2.4) * MM                 # mounting-hole offset from the board centre
        P = []                                         # parts
        add = P.append
        self._fasteners = []
        self._reliefs = []                             # (point, axis, radius, length): things on the middle frame that the lift-off cap must clear
        h_pcb = self.can_h                             # heights above the transducer-face plane, outward
        h_back = self.can_h + self.pcb_t
        h_out = self.can_h + self.lip                  # outer surface of the frame plates
        boards = []

        # ---------------------------------------------------------- frame: one plate per face
        def split_pieces(pid, spec, verts_out, cap=False):
            """Printed frame = three bodies: the lift-off cap (the six upper boards and the top port), the middle ring
            (upper halves of the equatorial plates, camera ports) and the lower bowl."""
            z = np.array(verts_out)[:, 2]
            if z.min() < -1e-6 and z.max() > 1e-6:
                for half in ("top", "bottom"):
                    s2 = dict(spec, id=f"{pid}.{half}", ops=spec["ops"] + [dict(op="half", keep=half)],
                              body="frame_mid" if half == "top" else "frame_lower")
                    add(s2)
            else:
                add(dict(spec, id=pid, body="frame_cap" if (cap or z.min() > 1e-6) else "frame_lower"))

        for f in G["sq"]:
            n, u, v, c = f["n"], f["u"], f["v"], f["c"]
            at = lambda x, y, h, c=c, u=u, v=v, n=n: c + x * u + y * v + h * n
            tag = f["name"]
            w2, cc = 26.25 * MM, 49.64 * MM            # window half-width and corner chamfer |x| + |y| <= cc
            k = cc - w2
            octa = [(w2, -k), (w2, k), (k, w2), (-k, w2), (-w2, k), (-w2, -k), (-k, -w2), (k, -w2)]
            pk = side / 2 + 0.15 * MM
            ops = [dict(op="prism", base=[_L(at(x, y, -1 * MM)) for x, y in octa], vec=_L(n * (h_out + 2 * MM))),
                   dict(op="prism", base=[_L(at(sx * pk, sy * pk, h_pcb)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))], vec=_L(n * 3 * MM))]
            ops += [dict(op="cyl", p=_L(at(sx * hm, sy * hm, h_pcb - 6.6 * MM)), axis=_L(n), r=1.02 * MM, h=8 * MM) for sx in (-1, 1) for sy in (-1, 1)]
            spec = dict(group="frame", kind="solid", shape="frustum", label=f"frame plate {tag} (window, board pocket, 4 screw holes)",
                        v_in=[_L(p) for p in f["verts"]], v_out=[_L(p * fo) for p in f["verts"]], ops=ops, color=COL["frame"], mat="plastic")
            split_pieces(f"{tag}.plate", spec, f["verts"] * fo)
            plate_ids = [f"{tag}.plate"] if abs(c[2]) > 1e-6 else [f"{tag}.plate.top", f"{tag}.plate.bottom"]
            # board, its back-side parts (one envelope for the interference check), screws, transducers
            holes = [drill(at(sx * hm, sy * hm, h_pcb), n, 1.1 * MM) for sx in (-1, 1) for sy in (-1, 1)]
            add(box(f"{tag}.pcb", "boards", f"TB-25 board {tag}", at(0, 0, h_pcb + self.pcb_t / 2), [side, side, self.pcb_t], COL["pcb"], x=u, z=n,
                    holes=holes, mat="pcb", mount=mount(plate_ids[0], "4x M2x8 thread-forming screw, seated in the frame pocket")))
            add(box(f"{tag}.parts", "board_parts", "board electronics (RP2040, 13 drivers, microphone, connectors)",
                    at(0, 0, h_back + self.back_h / 2), [side - 0.4 * MM, side - 0.4 * MM, self.back_h], "#2b2f36", x=u, z=n,
                    holes=[drill(at(sx * hm, sy * hm, h_back), n, 2.4 * MM) for sx in (-1, 1) for sy in (-1, 1)],
                    hidden=True, mount=mount(f"{tag}.pcb", "soldered")))
            for q, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                add(cyl(f"{tag}.screw{q}", "boards", "M2x8 screw", at(sx * hm, sy * hm, h_back), n, 1.0 * MM, 8.0 * MM, COL["steel"], mat="metal",
                        mount=mount(plate_ids[0] if len(plate_ids) == 1 else (plate_ids[0] if at(sx * hm, sy * hm, 0)[2] > 0 else plate_ids[1]), "threaded into the frame")))
                add(cyl(f"{tag}.screw{q}.head", "boards", "M2 screw head", at(sx * hm, sy * hm, h_back + 1.3 * MM), n, 1.9 * MM, 1.3 * MM, COL["steel"], mat="metal",
                        mount=mount(f"{tag}.pcb", "clamps the board")))
            self._fasteners += [dict(id=f"{tag}.screw{q}", size="M2", L=8, name="M2x8 thread-forming screw", thumb=False, insert=False, through=[], into=None, board=True) for q in range(4)]
            boards.append(dict(name=tag, c=_L(at(0, 0, h_back)), u=_L(u), v=_L(v), n=_L(n), cap=bool((f["verts"][:, 2] > 1e-6).all())))
        for j, (p, nrm) in enumerate(zip(arr.pos, arr.nrm)):
            tag = arr.panels[arr.panel_id[j]].name
            add(cyl(f"T{j:03d}", "transducers", "40 kHz transducer", p, nrm, self.can_d / 2, self.can_h, COL["can"], mat="metal",
                    mount=mount(f"{tag}.pcb", "soldered (2 leads)")))

        # ---------------------------------------------------------- ports
        cams, lights = [], []
        Z = np.array([0.0, 0.0, 1.0])
        ring = lambda cz, r, n=48: [_L([r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), cz]) for i in range(n)]

        def port_plate(t, pid, label, ops, cap=False):
            spec = dict(group="frame", kind="solid", shape="frustum", label=label, v_in=[_L(p) for p in t["verts"]],
                        v_out=[_L(p * fo) for p in t["verts"]], ops=ops, color=COL["frame"], mat="plastic")
            split_pieces(pid, spec, t["verts"] * fo, cap=cap)

        def emitter(t, pid, rho, j, body, kind, host):
            """405 nm LED on a 20 mm star with a 10 mm lens, on a seat that aims it at the build centre."""
            d = t["d"]
            q = d * rt * fo + rho * t["w"][j]                      # on the outer surface of the port plate
            ax = unit(-q)                                         # towards the build centre
            s = q - ax * 3.0 * MM                                 # centre of the star's front face
            add(cyl(f"{pid}.seat", "frame", "emitter seat", s, -ax, 9.8 * MM, 7.0 * MM, COL["frame"], bore=5.6 * MM, body=body, mat="plastic"))
            add(cyl(f"{pid}.star", "uv", "405 nm LED on 20 mm star board (SST-10-UV)", s - ax * 1.6 * MM, -ax, 9.6 * MM, 1.6 * MM, COL["alu"], mat="metal",
                    mount=mount(f"{pid}.seat", "2x M3x6 screw + thermal pad")))
            add(cyl(f"{pid}.lens", "uv", "10 mm lens (Ledil LISA2)", s, -ax, 4.95 * MM, 6.8 * MM, "#d9c8ff", mat="glass",
                    mount=mount(f"{pid}.star", "lens pins + adhesive")))
            sv = perp(ax); sw = np.cross(ax, sv)                  # wires solder at +-sv; the screws sit at +-sw
            for kk, sg in enumerate((-1, 1)):
                self._screw(P, f"{pid}.s{kk}", "M3", 6, s - ax * 1.6 * MM + sg * sw * 7.8 * MM, ax, through=[f"{pid}.star"], into=body, grip=1.6)
            lights.append(dict(kind=kind, id=pid, pos=_L(s + ax * 6.8 * MM), dir=_L(ax)))
            self._reliefs.append((s - ax * 2.4 * MM, ax, 11.0 * MM, 10.0 * MM))
            return dict(op="cyl", p=_L(s - ax * 2 * MM), axis=_L(ax), r=5.6 * MM, h=30 * MM)

        port_info = {}
        # ---- top port: conical seat; the build-pin chuck drops into it and comes out with the part
        t = G["ports"]["top"]
        zt = rt * fo
        port_plate(t, "port_top.plate", "frame plate, top port (conical seat for the build-pin chuck)",
                   [dict(op="cyl", p=[0, 0, rt - 1 * MM], axis=[0, 0, 1], r=self.port_d / 2, h=t_tri + 2 * MM),
                    dict(op="cone", p=[0, 0, zt - 5 * MM], axis=[0, 0, 1], r0=self.port_d / 2, r1=self.port_d / 2 + 1.0 * MM, h=5.02 * MM)], cap=True)
        z_rod_top = self.pole_tip_z + self.pole_len
        vb = 0.2 * MM * np.sqrt(2)                                 # the V-bore is offset so that the clamped pin sits on the machine axis
        add(dict(id="chuck.cone", group="top", kind="solid", shape="frustum", label="build-pin chuck: cone that centres it in the port (printed)", v_in=ring(zt - 5 * MM, self.port_d / 2 - 0.07 * MM),
                 v_out=ring(zt, self.port_d / 2 + 0.93 * MM), ops=[dict(op="cyl", p=[0, 0, zt - 6 * MM], axis=[0, 0, 1], r=3 * MM, h=8 * MM)], color=COL["top"], mat="plastic", body="chuck"))
        add(cyl("chuck.disc", "top", "build-pin chuck: flange", [0, 0, zt + 5 * MM], Z, 20 * MM, 5 * MM, COL["top"], bore=3 * MM, body="chuck", mat="plastic"))
        col_h = z_rod_top - zt - 5 * MM
        sq = 1.45 * MM * np.sqrt(2)
        add(cyl("chuck.column", "top", "build-pin chuck: column with a V-shaped bore; the thumbscrew presses the pin into the V", [0, 0, z_rod_top], Z, 8 * MM, col_h, COL["top"], body="chuck", mat="plastic",
                ops=[dict(op="prism", base=[[vb + sq, 0, zt + 4 * MM], [vb, sq, zt + 4 * MM], [vb - sq, 0, zt + 4 * MM], [vb, -sq, zt + 4 * MM]], vec=[0, 0, col_h + 2 * MM])]))
        add(cyl("chuck.stop", "top", "build-pin chuck: end stop (sets the pin height)", [0, 0, z_rod_top + 3 * MM], Z, 8 * MM, 3 * MM, COL["top"], body="chuck", mat="plastic"))
        add(cyl("pin", "pole", "build pin: 2.5 mm x 100 mm stainless rod (disposable)", [0, 0, z_rod_top], Z, self.pole_d / 2, self.pole_len, COL["steel"], mat="metal",
                mount=mount("chuck.column", "pressed into the V of the bore by the thumbscrew; its top end rests against the stop")))
        zc_ = zt + 5 * MM + 11 * MM
        self._screw(P, "chuck.thumb", "M3", 6.7, [8.0 * MM, 0, zc_], [-1, 0, 0], into="chuck.column", grip=0.0, insert=True, thumb=True, by="threaded through a heat-set insert; its tip bears on the pin")
        for kk, azv in enumerate(sorted(np.arctan2(v[1], v[0]) for v in t["verts"])[1:]):          # two ears towards two corners of the port plate
            e = np.array([np.cos(azv), np.sin(azv), 0.0])
            add(box(f"chuck.ear{kk}", "top", "build-pin chuck: ear", e * 24 * MM + Z * (zt + 2.5 * MM), [14 * MM, 9 * MM, 5 * MM], COL["top"], x=e, z=Z, body="chuck", mat="plastic"))
            self._screw(P, f"chuck.hold{kk}", "M3", 12, e * 26 * MM + Z * (zt + 5 * MM), -Z, through=[f"chuck.ear{kk}"], into="port_top.plate", grip=5.0, insert=True, thumb=True)
        port_info["chuck"] = mount("frame_cap", "sits in the conical seat of the top port; 2 M3 thumb screws; lifts out with the pin and the part")

        # ---- bottom port: opening for the jet and for drips; three bosses carry the legs and the cartridge dock
        t = G["ports"]["bottom"]
        zb_o = -rt * fo                                            # outer (lower) surface of the bottom plate
        vaz = sorted(float(np.arctan2(v[1], v[0])) for v in t["verts"])                          # corner directions: front-left, front-right, back
        E3 = [np.array([np.cos(a_), np.sin(a_), 0.0]) for a_ in vaz]
        T3 = [np.cross(Z, e) for e in E3]
        r_boss = 31 * MM
        ops = [dict(op="cyl", p=[0, 0, -rt + 1 * MM], axis=[0, 0, -1], r=self.port_d / 2, h=t_tri + 2 * MM),
               dict(op="prism", base=[[-1.5 * MM, -14 * MM, zb_o + 1 * MM], [1.5 * MM, -14 * MM, zb_o + 1 * MM], [1.5 * MM, -22 * MM, zb_o + 1 * MM], [-1.5 * MM, -22 * MM, zb_o + 1 * MM]],
                    vec=[0, 0, -2 * MM])]                             # groove that lets the nozzle tip slide out to the front
        port_plate(t, "port_bottom.plate", "frame plate, bottom port (jet comes up, drips fall through)", ops)
        z_dock = zb_o - 10 * MM                                    # underside of the bosses = top of the dock
        z_pad = z_dock - 4 * MM                                    # top of the leg pads
        tip = np.array([0.0, 0.0, self.nozzle_tip_z])
        feet = []
        for j, (e, tg) in enumerate(zip(E3, T3)):
            Bj = e * r_boss
            add(cyl(f"frame.boss{j}", "frame", "frame: boss for a leg (M4 insert)", Bj + Z * (zb_o + 0.6 * MM), Z, 5.6 * MM, 10.6 * MM, COL["frame"], body="frame_lower", mat="plastic"))
            # leg: a flat web in the radial plane, with a pad under the boss and a foot that goes through the floor foam
            z_foam = -self.enc_clear_bot
            z_floor = z_foam - self.foam_t
            prof = [(35, z_pad), (50, z_pad), (104, z_foam + 6 * MM), (104, z_foam - 8 * MM), (90, z_foam - 8 * MM), (90, z_foam + 1 * MM), (82, z_foam + 1 * MM), (35, z_pad - 8 * MM)]
            add(dict(id=f"leg{j}.web", group="stand", kind="solid", shape="prism", label="leg: web", base=[_L(e * rho * MM + Z * zz - tg * 4 * MM) for rho, zz in prof],
                     vec=_L(tg * 8 * MM), color=COL["stand"], mat="plastic", body=f"stand_leg{j}"))
            add(box(f"leg{j}.pad", "stand", "leg: pad under the frame boss", Bj + e * 0.5 * MM + Z * (z_pad - 4 * MM), [15 * MM, 14 * MM, 8 * MM], COL["stand"], x=e, z=Z, body=f"stand_leg{j}", mat="plastic"))
            foot_xy = e * 98 * MM
            add(cyl(f"leg{j}.foot", "stand", "leg: foot, through the floor foam onto the enclosure floor", foot_xy + Z * (z_foam + 6 * MM), Z, 11.5 * MM, 6 * MM + self.foam_t, COL["stand"], body=f"stand_leg{j}", mat="plastic"))
            self._screw(P, f"leg{j}.m4", "M4", 20, Bj + Z * (z_pad - 8 * MM), Z, through=[f"leg{j}.pad", "dock.plate"], into=f"frame.boss{j}", grip=12.0, insert=True)
            self._screw(P, f"leg{j}.m5", "M5", 20, foot_xy + Z * (z_floor - self.shell_t), Z, through=["enc.floor"], into=f"leg{j}.foot", grip=self.shell_t / MM, insert=True,
                        by="from inside the drawer, through the floor, into a heat-set insert in the foot")
            port_info[f"stand_leg{j}"] = mount("enc.floor", "one M5 bolt from inside the drawer into an insert in the foot; the foot passes through a hole in the floor foam")
            A_, B_ = e * 50 * MM + Z * z_pad, e * 104 * MM + Z * (z_foam + 6 * MM)
            feet.append(dict(e=e, az=float(np.arctan2(e[1], e[0])), xy=foot_xy[:2].copy(), A=A_, B=B_))
        # cartridge dock: a horseshoe plate clamped between the bosses and the leg pads; its slot opens to the front
        hull = []
        for e in E3:
            a0 = np.arctan2(e[1], e[0])
            hull.append([e[:2] * r_boss + 8 * MM * np.array([np.cos(a0 + da), np.sin(a0 + da)]) for da in (-np.pi / 3, 0, np.pi / 3)])
        fl, fr, bk = hull                                           # front-left, front-right, back
        sw_, yf = 13 * MM, float(min(fl[2][1], fr[0][1]))
        slot = [(-sw_, yf), (-sw_, 9 * MM), (-9.2 * MM, 13.2 * MM), (-2.5 * MM, 13.2 * MM), (-2.5 * MM, 20 * MM), (2.5 * MM, 20 * MM), (2.5 * MM, 13.2 * MM), (9.2 * MM, 13.2 * MM), (sw_, 9 * MM), (sw_, yf)]
        outline = [tuple(fl[2])] + slot + [tuple(p) for p in fr] + [tuple(p) for p in bk] + [tuple(fl[0]), tuple(fl[1])]
        add(dict(id="dock.plate", group="stand", kind="solid", shape="prism", label="cartridge dock (printed): the droplet generator slides in from the front", base=[[float(x), float(y), z_dock] for x, y in outline],
                 vec=[0, 0, -4 * MM], color=COL["pod"], mat="plastic", body="dock"))
        port_info["dock"] = mount("stand_leg0", "sits on the three leg pads; clamped between them and the frame bosses by the M4 leg screws")
        port_info["frame_lower"] = mount("dock", "3 M4 screws through the leg pads and the dock into heat-set inserts in the frame bosses")

        # ---- droplet generator cartridge (a piezo drop-on-demand generator of the Harris-lab type, turned nozzle-up)
        zf = z_dock                                                # the cartridge flange rests on the dock
        z_ch = zb_o - 27.5 * MM                                    # top of the pressure chamber, below the leg screws
        z_blk = self.nozzle_tip_z - 5 * MM                         # top of the heater block
        add(cyl("jet.flange", "jet", "droplet generator: flange that rests on the dock; also the drip tray", [0, 0, zf + 4 * MM], Z, 19 * MM, 4 * MM, COL["stand"], body="jet_body", mat="plastic",
                cuts=[dict(center=[0, 0, zf + 2 * MM], size=[17 * MM, 25 * MM, 6 * MM])]))
        add(cyl("jet.rim", "jet", "droplet generator: rim of the drip tray", [0, 0, zb_o - 3 * MM], Z, 19 * MM, 3 * MM, COL["stand"], bore=17.5 * MM, body="jet_body", mat="plastic"))
        for kk, sx in enumerate((-1, 1)):
            add(box(f"jet.wall{kk}", "jet", "droplet generator: wall joining flange and chamber through the dock slot", [sx * 10.5 * MM, 0, 0.5 * (zf + 2 * MM + z_ch - 2 * MM)],
                    [4 * MM, 16 * MM, zf + 2 * MM - (z_ch - 2 * MM)], COL["stand"], body="jet_body", mat="plastic"))
        add(cyl("jet.top", "jet", "droplet generator: chamber top", [0, 0, z_ch], Z, 24 * MM, 4 * MM, COL["stand"], body="jet_body", mat="plastic", holes=[drill([0, 0, z_ch], Z, 3.05 * MM)]))
        add(cyl("jet.funnel", "jet", "droplet generator: pressure chamber (28.6 mm bore)", [0, 0, z_ch - 4 * MM], Z, 24 * MM, 10 * MM, COL["stand"], bore=14.3 * MM, body="jet_body", mat="plastic"))
        add(cyl("jet.piezo", "jet", "piezo buzzer disc, 35 mm (CUI CEB-35D26): its brass face is the chamber wall", [0, 0, z_ch - 14 * MM], Z, 17.5 * MM, 0.6 * MM, COL["brass"], mat="metal",
                mount=mount("jet.funnel", "sealed on an O-ring, clamped by the ring below")))
        add(cyl("jet.clamp", "jet", "droplet generator: clamp ring", [0, 0, z_ch - 14.6 * MM], Z, 24 * MM, 6 * MM, COL["pod"], bore=12 * MM, body="jet_clamp", mat="plastic"))
        add(cyl("jet.clamp.rim", "jet", "droplet generator: clamp ring, rim round the piezo disc", [0, 0, z_ch - 14 * MM], Z, 24 * MM, 0.6 * MM, COL["pod"], bore=17.7 * MM, body="jet_clamp", mat="plastic"))
        for kk in range(6):
            ang = np.radians(30 + 60 * kk)
            self._screw(P, f"jet.cs{kk}", "M3", 16, [20.5 * MM * np.cos(ang), 20.5 * MM * np.sin(ang), z_ch - 20.6 * MM], Z, through=["jet_clamp"], into="jet.funnel", grip=6.6, insert=True)
        port_info["jet_clamp"] = mount("jet_body", "6 M3x16 screws into heat-set inserts, tightened evenly")
        add(box("jet.block", "jet", "heater block (3-D printer hot-end type, M6 thread) with heater cartridge and thermistor", [0, 0, z_blk - 5.75 * MM], [16 * MM, 23 * MM, 11.5 * MM], COL["alu"], mat="metal",
                mount=mount("jet.break", "screwed onto the feed tube")))
        add(cyl("jet.break", "jet", "feed tube (hot-end heat break, M6): carries the resin from the chamber to the nozzle", [0, 0, z_blk - 11.5 * MM], Z, 3.0 * MM, z_blk - 11.5 * MM - (z_ch - 4.2 * MM), COL["steel"], mat="metal",
                mount=mount("jet.top", "pushed into the chamber top with an O-ring, locked by an M3 grub screw")))
        add(cyl("jet.nozzle", "jet", "nozzle: brass M6 3-D printer nozzle, 0.30 mm bore", [0, 0, z_blk + 3 * MM], Z, 4 * MM, 3 * MM, COL["brass"], mat="metal", mount=mount("jet.block", "M6 thread")))
        add(cyl("jet.tip", "jet", "nozzle tip", tip, Z, 1.0 * MM, 2 * MM, COL["brass"], mat="metal", mount=mount("jet.nozzle", "one piece")))
        add(cyl("jet.barb", "jet", "hose barb (resin inlet)", [0, 30 * MM, z_ch - 9 * MM], [0, 1, 0], 2 * MM, 6.0 * MM, COL["brass"], mat="metal", mount=mount("jet.funnel", "10-24 thread with a washer")))
        self._screw(P, "jet.lock", "M3", 8, [5 * MM, 16.8 * MM, z_dock - 4 * MM], Z, through=["dock.plate"], into="jet.flange", grip=4.0, thumb=True, by="threaded into the flange: holds the cartridge in the dock")
        port_info["jet_body"] = mount("dock", "slides onto the dock from the front and rests on it; locked by one M3 thumb screw from below")
        lights.append(dict(kind="jet", id="nozzle", pos=_L(tip), dir=_L(Z)))
        # ---- droplet light gate on a bracket screwed to the bottom port plate; its slot faces the front
        eb, tb = E3[2], T3[2]
        zg = self.nozzle_tip_z + 4 * MM
        add(box("gate.foot", "gate", "light-gate bracket: foot", eb * 16.4 * MM + Z * (zb_o - 1.25 * MM), [16.8 * MM, 8 * MM, 2.5 * MM], COL["pod"], x=eb, z=Z, body="gate_bracket", mat="plastic"))
        add(box("gate.riser", "gate", "light-gate bracket: riser inside the port", eb * 11.2 * MM + Z * 0.5 * (zg + 3.15 * MM + zb_o - 2.5 * MM), [6.4 * MM, 8 * MM, zg + 3.15 * MM - (zb_o - 2.5 * MM)], COL["pod"], x=eb, z=Z,
                body="gate_bracket", mat="plastic"))
        add(box("gate.base", "gate", "droplet light gate (TCST2300 photo-interrupter)", eb * 6.75 * MM + Z * zg, [2.5 * MM, 24.5 * MM, 6.3 * MM], COL["black"], x=eb, z=Z, mat="plastic",
                mount=mount("gate.riser", "pressed onto two printed pins of the riser and bonded")))
        for sgn in (-1, 1):
            add(box(f"gate.post{sgn}", "gate", "light gate, emitter / detector post", eb * 1.35 * MM + tb * sgn * 3.75 * MM + Z * zg, [8.3 * MM, 4.4 * MM, 6.3 * MM], COL["black"], x=eb, z=Z, mat="plastic",
                    mount=mount("gate.base", "one piece")))
        self._screw(P, "gate.s0", "M2.5", 8, eb * 22.2 * MM + Z * (zb_o - 2.5 * MM), Z, through=["gate.foot"], into="port_bottom.plate", grip=2.5, insert=True)
        port_info["gate_bracket"] = mount("frame_lower", "one M2.5 screw into a heat-set insert in the bottom port plate")
        # ---- resin reservoir: an open syringe barrel on a height-adjustable clamp on the back leg; its level sets the meniscus
        f2 = feet[2]
        rc = f2["e"] * 62 * MM + T3[2] * 10 * MM + Z * (z_pad - 16 * MM)
        add(box("res.clamp", "jet", "reservoir clamp (printed): slides up and down to set the resin level about 10 mm below the nozzle", rc, [16 * MM, 12 * MM, 20 * MM], COL["pod"], x=f2["e"], z=Z, body="res_clamp", mat="plastic"))
        add(cyl("res.barrel", "jet", "resin reservoir: 10 mL syringe barrel, open top, light-tight cap", rc + T3[2] * 14 * MM + Z * 28 * MM, Z, 8 * MM, 56 * MM, COL["resin"], mat="glass",
                mount=mount("res.clamp", "clipped in; height set by the clamp")))
        for kk, dz in enumerate((-5, 5)):
            self._screw(P, f"res.s{kk}", "M3", 14, rc - T3[2] * 14 * MM + Z * dz * MM, T3[2], through=["leg2.web"], into="res.clamp", grip=8.0, insert=True)
        port_info["res_clamp"] = mount("stand_leg2", "2 M3x14 screws through the back leg into heat-set inserts in the clamp")

        # ---- side ports
        cam_len, holder_h = 22 * MM, 11 * MM
        cap_lugs = []
        for q, t in enumerate(G["ports"]["upper"]):
            d, w = t["d"], t["w"]
            pid = f"cam{q}"
            ops = [dict(op="cyl", p=_L(d * (rt - 1 * MM)), axis=_L(d), r=7.6 * MM, h=t_tri + 2 * MM)]
            body = "frame_mid"
            host = f"port_u{q}.plate.top"
            ops.append(emitter(t, f"uv_u{q}a", 21.8 * MM, 1, body, "uv", host))
            if q == 0:
                qq = d * rt * fo + 23 * MM * w[2]
                ax = unit(-qq)
                front = qq + ax * 4 * MM
                add(cyl("laser.seat", "frame", "line-laser seat", qq - ax * 10 * MM, -ax, 8 * MM, 14 * MM, COL["frame"], bore=5.06 * MM, body=body, mat="plastic"))
                add(cyl("laser", "scanner", "line laser 650 nm 5 mW (10 x 33 mm)", front - ax * 32.8 * MM, -ax, 5.06 * MM, 32.8 * MM, COL["brass"], mat="metal",
                        mount=mount("laser.seat", "push fit + M3 grub screw")))
                ops.append(dict(op="cyl", p=_L(qq - ax * 2 * MM), axis=_L(ax), r=5.06 * MM, h=30 * MM))
                self._reliefs.append((qq - ax * 10.5 * MM, ax, 8.6 * MM, 15 * MM))
                lights.append(dict(kind="laser", id="laser", pos=_L(front), dir=_L(ax)))
            else:
                ops.append(emitter(t, f"uv_u{q}b", 21.8 * MM, 2, body, "uv", host))
            port_plate(t, f"port_u{q}.plate", "frame plate, camera port", ops)
            # camera pod: barrel on the port plate, flange that the camera board screws to
            r_lens = rt + 0.5 * MM                                 # lens front, just inside the frame
            r_pcb = r_lens + cam_len + 2.5 * MM                    # front face of the camera board
            r_out = rt * fo
            e1 = w[0]                                              # sensor diagonal along the vertical: the pod is rolled 45 deg
            e2 = np.cross(d, e1)
            ex, ey = unit(e1 + e2), unit(e2 - e1)
            add(cyl(f"{pid}.pod", "pods", "camera pod (printed)", d * (r_pcb - 3.5 * MM), d, 11.4 * MM, r_pcb - 3.5 * MM - r_out, COL["pod"], bore=9.3 * MM, body=f"pod{q}", mat="plastic"))
            add(cyl(f"{pid}.pod.collar", "pods", "camera pod collar", d * (r_pcb - 3.5 * MM), d, 15 * MM, 3 * MM, COL["pod"], bore=9.3 * MM, body=f"pod{q}", mat="plastic"))
            add(box(f"{pid}.pod.flange", "pods", "camera pod flange", d * (r_pcb - 1.75 * MM), [36 * MM, 36 * MM, 3.5 * MM], COL["pod"], x=ex, z=d, body=f"pod{q}", mat="plastic",
                    holes=[drill(d * r_pcb, d, 12.2 * MM)]))
            for jj in range(3):
                en = unit(-(w[jj]))                                # towards the middle of an edge
                add(cyl(f"{pid}.pod.lug{jj}", "pods", "camera pod lug", d * (r_out + 3 * MM) + en * 14.0 * MM, d, 3 * MM, 3 * MM, COL["pod"], body=f"pod{q}", mat="plastic"))
                self._screw(P, f"{pid}.pod.s{jj}", "M2.5", 8, d * (r_out + 3 * MM) + en * 14.0 * MM, -d, through=[f"{pid}.pod.lug{jj}"], into=host, grip=3.0, insert=True)
            port_info[f"pod{q}"] = mount("frame_mid", "3 M2.5x8 screws through the lugs into heat-set inserts in the port plate")
            add(box(f"{pid}.pcb", "camera", "global-shutter camera board (OV9281, 32 x 32 mm)", d * (r_pcb + 0.8 * MM), [32 * MM, 32 * MM, 1.6 * MM], COL["pcb_black"], x=ex, z=d, mat="pcb",
                    mount=mount(f"{pid}.pod.flange", "4 M2x5 screws")))
            for jj, (sa, sb) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                self._screw(P, f"{pid}.pcb.s{jj}", "M2", 5, d * (r_pcb + 1.6 * MM) + (sa * ex + sb * ey) * 14 * MM, -d, through=[f"{pid}.pcb"], into=f"{pid}.pod.flange", grip=1.6)
            add(cyl(f"{pid}.holder", "camera", "M12 lens holder", d * r_pcb, d, 9 * MM, holder_h, COL["black"], bore=7 * MM, mat="plastic", mount=mount(f"{pid}.pcb", "part of the camera module")))
            add(cyl(f"{pid}.lens", "camera", "M12 lens, 2.8 mm", d * (r_pcb - 2.5 * MM), d, 7 * MM, cam_len, COL["lens"], mat="metal", mount=mount(f"{pid}.holder", "M12x0.5 thread + lock ring")))
            add(box(f"{pid}.usb", "camera", "camera USB connector", d * (r_pcb + 1.6 * MM + 2 * MM) + ey * 11 * MM, [8 * MM, 5 * MM, 4 * MM], COL["white"], x=ex, z=d, mat="plastic",
                    mount=mount(f"{pid}.pcb", "soldered")))
            cams.append(dict(id=pid, pos=_L(d * (r_lens + 4 * MM)), dir=_L(-d), up=_L(-e1), f_mm=self.cam_f_mm, sensor_mm=list(self.cam_sensor_mm), px=list(self.cam_px),
                             roll_deg=45.0, port=_L(d), usb=_L(d * (r_pcb + 1.6 * MM + 4 * MM) + ey * 11 * MM), trig=_L(d * (r_pcb + 1.6 * MM) - ey * 12 * MM)))
        for q, t in enumerate(G["ports"]["lower"]):
            d, w = t["d"], t["w"]
            r_out = rt * fo
            ops = [dict(op="cyl", p=_L(d * (rt - 1 * MM)), axis=_L(d), r=8 * MM, h=t_tri + 2 * MM),
                   dict(op="cyl", p=_L(d * (r_out - 3.1 * MM)), axis=_L(d), r=10.2 * MM, h=5 * MM)]
            body = "frame_lower"
            host = f"port_l{q}.plate.bottom"
            for jj, sfx in ((1, "a"), (2, "b")):
                ops.append(emitter(t, f"uv_l{q}{sfx}", 21.8 * MM, jj, body, "uv", host))
            port_plate(t, f"port_l{q}.plate", "frame plate, strobe port", ops)
            add(cyl(f"strobe{q}.diffuser", "strobe", "opal acrylic diffuser, 20 mm", d * r_out, d, 10 * MM, 3 * MM, "#f4f1ea", mat="glass",
                    mount=mount("frame_lower", "glued into the counterbore of the port plate")))
            add(cyl(f"strobe{q}.ring", "frame", "strobe spacer ring", d * (r_out + 4 * MM), d, 11.5 * MM, 4 * MM, COL["frame"], bore=6.0 * MM, body=body, mat="plastic"))
            add(cyl(f"strobe{q}.star", "strobe", "red LED on 20 mm star board (Cree XP-E2)", d * (r_out + 5.6 * MM), d, 9.6 * MM, 1.6 * MM, COL["alu"], mat="metal",
                    mount=mount(f"strobe{q}.ring", "2 M3x6 screws")))
            sv = perp(d); sw = np.cross(d, sv)
            for kk, sg in enumerate((-1, 1)):
                self._screw(P, f"strobe{q}.s{kk}", "M3", 5, d * (r_out + 5.6 * MM) + sg * sw * 8.0 * MM, -d, through=[f"strobe{q}.star"], into=body, grip=1.6)
            lights.append(dict(kind="strobe", id=f"strobe{q}", pos=_L(d * (r_out - 0.5 * MM)), dir=_L(-d)))

        # middle and lower frame: joined across the equator by a screw at the tip of each side port (and by the six equatorial boards)
        for q, t in enumerate(G["ports"]["upper"] + G["ports"]["lower"]):
            d, w0 = t["d"], t["w"][0]
            c_out = d * rt * fo
            s0 = -c_out[2] / w0[2]
            p0 = c_out + s0 * w0
            dh = unit(np.append(d[:2], 0))
            th = np.cross([0, 0, 1.0], dh)
            cen = p0 + dh * 4.6 * MM
            sc = cen + dh * 2.2 * MM
            for sfx, body, z0, z1 in (("u", "frame_mid", 0.0, 6 * MM), ("l", "frame_lower", -6 * MM, 0.0)):
                add(box(f"tab{q}.{sfx}", "frame", "frame joining tab", cen + np.array([0, 0, 0.5 * (z0 + z1)]), [8 * MM, 11.5 * MM, z1 - z0], COL["frame"], x=th, z=[0, 0, 1], body=body, mat="plastic"))
                add(cyl(f"tab{q}.{sfx}.round", "frame", "rounded end of the tab", cen + dh * 5.75 * MM + np.array([0, 0, z1]), Z, 4 * MM, z1 - z0, COL["frame"], body=body, mat="plastic"))
            self._screw(P, f"tab{q}.screw", "M3", 12, sc + np.array([0, 0, 6 * MM]), -Z, through=["frame_mid"], into="frame_lower", grip=6.0, insert=True)
        port_info["frame_mid"] = mount("frame_lower", "6 M3x12 screws at the joining tabs; the six equatorial boards bridge the joint")
        port_info["frame_cap"] = mount("frame_mid", "rests by its own weight in three V-notches of the middle frame, which locate it; no fastener - it lifts straight off")

        # air sensors on printed shelves: one on a camera pod (top of the cage), one on a leg (bottom)
        q_s = int(np.argmax([tt_["d"][1] for tt_ in G["ports"]["upper"]]))        # the camera port that faces the back wall
        t = G["ports"]["upper"][q_s]
        d, sd = t["d"], -t["w"][0]
        r_fl = rt + 0.5 * MM + cam_len + 2.5 * MM
        add(box("bme_a.shelf", "pods", "sensor shelf on the camera pod", d * (r_fl - 1.75 * MM) + sd * 36 * MM, [26.5 * MM, 20 * MM, 3.5 * MM], COL["pod"], x=sd, z=d, body=f"pod{q_s}", mat="plastic"))
        add(box("bme_a", "sensors", "BME280 air sensor (top)", d * (r_fl + 2.3 * MM) + sd * 37.5 * MM, [25.2 * MM, 18 * MM, 4.6 * MM], "#0b3d91", x=sd, z=d, mat="pcb",
                mount=mount("bme_a.shelf", "2 M2.5x6 screws")))
        s2 = np.cross(d, sd)
        for kk, sg in enumerate((-1, 1)):
            self._screw(P, f"bme_a.s{kk}", "M2.5", 8, d * (r_fl + 4.6 * MM) + sd * (37.5 + sg * 10) * MM, -d, through=["bme_a"], into="bme_a.shelf", grip=4.6)
        f0 = feet[0]
        mid = 0.55 * f0["A"] + 0.45 * f0["B"]
        add(box("bme_b.shelf", "stand", "leg: sensor shelf", mid - T3[0] * 12 * MM, [22 * MM, 30 * MM, 3 * MM], COL["stand"], x=f0["e"], z=[0, 0, 1], body="stand_leg0", mat="plastic"))
        add(box("bme_b", "sensors", "BME280 air sensor (bottom)", mid + np.array([0, 0, 3.8 * MM]) - T3[0] * 14 * MM, [18 * MM, 14 * MM, 4.6 * MM], "#0b3d91", x=f0["e"], z=[0, 0, 1], mat="pcb",
                mount=mount("bme_b.shelf", "2 M2.5x6 screws")))
        for kk, sg in enumerate((-1, 1)):
            self._screw(P, f"bme_b.s{kk}", "M2.5", 6, mid + np.array([0, 0, 6.1 * MM]) - T3[0] * 14 * MM + f0["e"] * sg * 6 * MM, -Z, through=["bme_b"], into="bme_b.shelf", grip=4.6)

        # ---------------------------------------------------------- enclosure, thermal, drawer
        P += self._enclosure(feet)
        names = dict(frame_cap="frame cap", frame_mid="middle frame", frame_lower="lower frame", chuck="build-pin chuck", dock="cartridge dock", jet_body="droplet-generator body",
                     jet_clamp="droplet-generator clamp ring", gate_bracket="light-gate bracket", res_clamp="reservoir clamp")
        for p in P:                                                # body-level fastenings
            if p.get("body") in port_info and "mount" not in p:
                m_ = port_info[p["body"]]
                nm = names.get(p["body"], "camera pod" if p["body"].startswith("pod") else "leg")
                p["mount"] = dict(to=m_["to"], by=f"one printed piece with the {nm}, which is held like this: " + m_["by"])
        self._cut_fastener_holes(P)
        wires = self._wires(G, brd, feet, cams, lights, boards)
        return dict(parts=P, wires=wires, cameras=cams, lights=lights, boards=boards, feet=[dict(az=f["az"], xy=_L(f["xy"])) for f in feet])

    # ------------------------------------------------------------------ enclosure
    def _enc_dims(self):
        a, zt, zb, f, sh = self.enc_clear_xy, self.enc_clear_top, self.enc_clear_bot, self.foam_t, self.shell_t
        return dict(a=a, zt=zt, zb=zb, f=f, sh=sh, b=a + f, zc=0.5 * (zt - zb), hz=0.5 * (zt + zb), z_sf=-zb - f - sh,
                    z_tray=-zb - f - sh - self.drawer_h, z_unit=-50 * MM, y_unit=50 * MM)

    def _pass_throughs(self, feet):
        """Holes for the harness: next to each foot, and one under the jet."""
        out = []
        for f in feet:
            az = f["az"] + np.radians(15)
            out.append(dict(xy=[float(108 * MM * np.cos(az)), float(108 * MM * np.sin(az))], r=11 * MM, name="harness"))
        out.append(dict(xy=[0.0, 0.03], r=7 * MM, name="services"))
        return out

    def _enclosure(self, feet):
        """Sealed, foam-lined, temperature-controlled chamber, the thermal units on its side walls, and the electronics drawer."""
        E = self._enc_dims()
        a, zt, zb, f, sh, b, zc, hz, z_sf, z_tray, zu = (E[k] for k in ("a", "zt", "zb", "f", "sh", "b", "zc", "hz", "z_sf", "z_tray", "z_unit"))
        yu = E["y_unit"]
        out = []
        add = out.append
        Z = [0, 0, 1]
        pts = self._pass_throughs(feet)
        fl_holes = [drill([p["xy"][0], p["xy"][1], 0], Z, p["r"]) for p in pts]
        foot_holes = [drill([ft["xy"][0], ft["xy"][1], 0], Z, 12.5 * MM) for ft in feet]
        full = 2 * b + 2 * sh
        add(box("enc.floor", "enclosure_shell", "enclosure floor (6 mm plywood)", [0, 0, -zb - f - sh / 2], [full, full, sh], COL["ply"], mat="wood", holes=fl_holes, root=True))
        add(box("foam.floor", "enclosure_foam", "melamine foam, floor", [0, 0, -zb - f / 2], [2 * b, 2 * b, f], COL["foam"], mat="foam", holes=fl_holes + foot_holes,
                mount=mount("enc.floor", "spray adhesive")))
        unit_cut_shell = lambda sx: dict(center=[sx * (b + sh / 2), yu, zu], size=[sh + 2 * MM, 42 * MM, 62 * MM])
        unit_cut_foam = lambda sx: dict(center=[sx * (a + f / 2), yu, zu], size=[f + 2 * MM, 125 * MM, 125 * MM])
        sw_cut = dict(center=[a + f / 2, -(a - 23 * MM), zt - 7 * MM], size=[f + 2 * MM, 46 * MM, 14 * MM])
        for sx, nm in ((-1, "left"), (1, "right")):
            add(box(f"enc.side_{nm}", "enclosure_shell", f"enclosure side, {nm}", [sx * (b + sh / 2), 0, zc], [sh, 2 * b, 2 * hz + 2 * f], COL["ply"], mat="wood",
                    cuts=[unit_cut_shell(sx)], mount=mount("enc.floor", "wood glue, with glue blocks in the corners")))
            add(box(f"foam.side_{nm}", "enclosure_foam", f"melamine foam, {nm} side", [sx * (a + f / 2), 0, zc], [f, 2 * a, 2 * hz], COL["foam"], mat="foam",
                    cuts=[unit_cut_foam(sx)] + ([sw_cut] if sx > 0 else []), mount=mount(f"enc.side_{nm}", "spray adhesive")))
        add(box("enc.back", "enclosure_shell", "enclosure back", [0, b + sh / 2, zc], [full, sh, 2 * hz + 2 * f], COL["ply"], mat="wood", mount=mount("enc.floor", "wood glue, with glue blocks in the corners")))
        add(box("foam.back", "enclosure_foam", "melamine foam, back", [0, a + f / 2, zc], [2 * b, f, 2 * hz], COL["foam"], mat="foam", mount=mount("enc.back", "spray adhesive")))
        win = dict(center=[0, -(b + sh / 2), 5 * MM], size=[170 * MM, sh + 2 * MM, 170 * MM])
        winf = dict(center=[0, -(a + f / 2), 5 * MM], size=[170 * MM, f + 2 * MM, 170 * MM])
        add(box("enc.door", "door", "door (plywood, gasketed, interlocked)", [0, -(b + sh / 2), zc], [full, sh, 2 * hz + 2 * f], COL["ply"], mat="wood", cuts=[win],
                mount=mount("hinge.door0", "2 hinges on the left side; magnetic latch on the right")))
        add(box("foam.door", "door", "melamine foam, door", [0, -(a + f / 2), zc], [2 * b, f, 2 * hz], COL["foam"], mat="foam", cuts=[winf], mount=mount("enc.door", "spray adhesive")))
        add(box("door.window", "door", "door window: 405 nm blocking acrylic, 3 mm", [0, -(b + sh + 1.6 * MM), 5 * MM], [200 * MM, 3.2 * MM, 200 * MM], COL["acrylic"], mat="acrylic",
                mount=mount("enc.door", "8 pan-head wood screws, foam gasket")))
        for kk, (wx, wz) in enumerate(((-92.5, -92.5), (0, -92.5), (92.5, -92.5), (92.5, 0), (92.5, 92.5), (0, 92.5), (-92.5, 92.5), (-92.5, 0))):
            self._screw(out, f"door.window.s{kk}", "3.5 mm", 9.5, [wx * MM, -(b + sh + 3.2 * MM), 5 * MM + wz * MM], [0, 1, 0], through=["door.window"], into="enc.door", grip=3.2, wood=True, group="door")
        add(box("enc.lid", "lid", "lid (lifts for part removal, interlocked)", [0, 0, zt + f + sh / 2], [full, full, sh], COL["ply"], mat="wood",
                mount=mount("hinge.lid0", "2 hinges on the back; gas strut optional")))
        add(box("foam.lid", "lid", "melamine foam, lid", [0, 0, zt + f / 2], [2 * b, 2 * b, f], COL["foam"], mat="foam", mount=mount("enc.lid", "spray adhesive")))
        for q, zz in enumerate((zc - 80 * MM, zc + 80 * MM)):
            yc = -(b + sh / 2) + 3 * MM
            add(box(f"hinge.door{q}", "hardware", "door hinge", [-(b + sh + 1.5 * MM), yc, zz], [3 * MM, 30 * MM, 30 * MM], COL["steel"], mat="metal",
                    mount=mount("enc.side_left", "2 wood screws into the side, 2 into the door edge")))
            for kk, dz in enumerate((-9 * MM, 9 * MM)):
                self._screw(out, f"hinge.door{q}.a{kk}", "3.5 mm", 9, [-(b + sh + 3 * MM), yc + 8 * MM, zz + dz], [1, 0, 0], through=[f"hinge.door{q}"], into="enc.side_left", grip=3.0, wood=True, group="hardware")
                self._screw(out, f"hinge.door{q}.b{kk}", "3.5 mm", 16, [-(b + sh + 3 * MM), yc - 3 * MM, zz + dz], [1, 0, 0], through=[f"hinge.door{q}"], into="enc.door", grip=3.0, wood=True, group="hardware")
        for q, xx in enumerate((-80 * MM, 80 * MM)):
            zh = zt + f + sh / 2 - 12 * MM
            add(box(f"hinge.lid{q}", "hardware", "lid hinge", [xx, b + sh + 1.5 * MM, zh], [30 * MM, 3 * MM, 30 * MM], COL["steel"], mat="metal",
                    mount=mount("enc.back", "2 wood screws into the back, 2 into the lid edge")))
            for kk, dx in enumerate((-9 * MM, 9 * MM)):
                self._screw(out, f"hinge.lid{q}.a{kk}", "3.5 mm", 9, [xx + dx, b + sh + 3 * MM, zh - 6 * MM], [0, -1, 0], through=[f"hinge.lid{q}"], into="enc.back", grip=3.0, wood=True, group="hardware")
                self._screw(out, f"hinge.lid{q}.b{kk}", "3.5 mm", 16, [xx + dx, b + sh + 3 * MM, zh + 12 * MM], [0, -1, 0], through=[f"hinge.lid{q}"], into="enc.lid", grip=3.0, wood=True, group="hardware")
        # interlock switches in a pocket of the right-hand foam: one faces the door, one faces the lid
        add(box("sw.door", "sensors", "door interlock switch (Omron SS-5GL)", [b - 3.2 * MM, -(a - 9.9 * MM), zt - 8.9 * MM], [6.4 * MM, 19.8 * MM, 10.2 * MM], COL["black"], mat="plastic",
                mount=mount("enc.side_right", "2x M2.3 screw; the lever touches the closed door")))
        add(box("sw.lid", "sensors", "lid interlock switch (Omron SS-5GL)", [b - 3.2 * MM, -(a - 32.7 * MM), zt - 5.1 * MM], [6.4 * MM, 19.8 * MM, 10.2 * MM], COL["black"], mat="plastic",
                mount=mount("enc.side_right", "2x M2.3 screw; the lever touches the closed lid")))
        for nm, yy, zz in (("sw.door", -(a - 9.9 * MM), zt - 8.9 * MM), ("sw.lid", -(a - 32.7 * MM), zt - 5.1 * MM)):
            for kk, dy in enumerate((-4.75 * MM, 4.75 * MM)):
                self._screw(out, f"{nm}.s{kk}", "M2", 10, [b - 6.4 * MM, yy + dy, zz], [1, 0, 0], through=[nm], into="enc.side_right", grip=6.4, group="hardware")
        # thermal units: Peltier assembly outside, its cold plate through the wall, a finned sink and a fan inside
        for sx, nm in ((1, "right"), (-1, "left")):
            w = f"enc.side_{nm}"
            x0 = sx * (b + sh)
            X = [sx, 0, 0]
            add(box(f"tec_{nm}.sink_out", "thermal", "Peltier assembly: heat sink + fan, outside (90 x 90 x 66 mm)", [x0 + sx * 33 * MM, yu, zu], [66 * MM, 90 * MM, 90 * MM], COL["alu"], mat="metal",
                    mount=mount(w, "4x M4 screw through the wall")))
            add(box(f"tec_{nm}.cold", "thermal", "Peltier cold plate (40 x 60 mm), through the wall", [x0 - sx * 6 * MM, yu, zu], [12 * MM, 40 * MM, 60 * MM], COL["alu"], mat="metal",
                    mount=mount(f"tec_{nm}.sink_out", "part of the Peltier assembly")))
            add(box(f"tec_{nm}.sink_in", "thermal", "inside heat sink (100 x 70 x 25 mm)", [x0 - sx * 24.5 * MM, yu, zu], [25 * MM, 100 * MM, 70 * MM], COL["alu"], mat="metal",
                    mount=mount(f"tec_{nm}.cold", "4x M3 screw + thermal paste")))
            for kk, (dy, dz) in enumerate(((-35, -35), (35, -35), (35, 35), (-35, 35))):
                self._screw(out, f"tec_{nm}.w{kk}", "M4", 16, [sx * b, yu + dy * MM, zu + dz * MM], X, through=[w], into=f"tec_{nm}.sink_out", grip=sh / MM, group="thermal")
            for kk, (dy, dz) in enumerate(((-15, -25), (15, -25), (15, 25), (-15, 25))):
                self._screw(out, f"tec_{nm}.c{kk}", "M3", 30, [x0 - sx * 37 * MM, yu + dy * MM, zu + dz * MM], X, through=[f"tec_{nm}.sink_in"], into=f"tec_{nm}.cold", grip=25.0, group="thermal")
            add(box(f"tec_{nm}.fan", "thermal", "inside fan, 120 mm (draws chamber air onto the sink)", [x0 - sx * 49.5 * MM, yu, zu], [25 * MM, 120 * MM, 120 * MM], COL["fan"], mat="plastic",
                    holes=[drill([x0 - sx * 49.5 * MM, yu, zu], X, 56 * MM)], mount=mount(f"tec_{nm}.sink_in", "4 printed clips")))
        # electronics drawer: a tray screwed under the floor; front panel comes off
        tw = 6.35 * MM
        add(box("drawer.floor", "electronics", "electronics drawer, base", [0, 0, z_tray - tw / 2], [full, full, tw], COL["ply"], mat="wood", mount=mount("drawer.wall_left", "wood glue")))
        for sx, nm in ((-1, "left"), (1, "right")):
            add(box(f"drawer.wall_{nm}", "electronics", f"electronics drawer, {nm} wall", [sx * (b + sh - tw / 2), 0, z_tray + self.drawer_h / 2], [tw, full, self.drawer_h], COL["ply"], mat="wood",
                    holes=[drill([sx * b, yy, z_tray + 36 * MM], [1, 0, 0], 5 * MM) for yy in (yu - 30 * MM, yu + 30 * MM)], mount=mount("enc.floor", "wood glue, with glue blocks in the corners")))
        add(box("drawer.wall_back", "electronics", "electronics drawer, back wall (cable glands, mains inlet)", [0, b + sh - tw / 2, z_tray + self.drawer_h / 2],
                [full - 2 * tw, tw, self.drawer_h], COL["ply"], mat="wood", mount=mount("enc.floor", "wood glue, with glue blocks in the corners")))
        add(box("drawer.front", "door", "electronics drawer, front panel (removable)", [0, -(b + sh - tw / 2), z_tray + self.drawer_h / 2],
                [full - 2 * tw, tw, self.drawer_h], COL["ply"], mat="wood", mount=mount("drawer.floor", "4 magnetic catches; pulls off without tools")))
        zt0 = z_tray

        def module(pid, label, x, y, sx, sy, sz, color, mat="plastic", lift=0.0, by=None, to="drawer.floor"):
            add(box(pid, "electronics", label, [x, y, zt0 + lift + sz / 2], [sx, sy, sz], color, mat=mat, mount=mount(to, by or "2 M3x10 screws from below, through the drawer base")))
            if by is None:
                for kk, sg in enumerate((-1, 1)):
                    self._screw(out, f"{pid}.s{kk}", "M3", 10, [x + sg * (sx / 2 - 8 * MM), y, zt0 - tw], [0, 0, 1], through=["drawer.floor"], into=pid, grip=tw / MM, group="electronics")

        module("psu24", "drive supply: Mean Well LRS-100-24, 24 V 4.5 A (129 x 97 x 30 mm)", -0.085, 0.100, 129 * MM, 97 * MM, 30 * MM, COL["alu"], "metal")
        module("psu12", "thermal supply: Mean Well LRS-150-12, 12 V 12.5 A (159 x 97 x 30 mm)", 0.075, 0.100, 159 * MM, 97 * MM, 30 * MM, COL["alu"], "metal")
        module("buck18", "adjustable step-down converter: sets the drive voltage (12-24 V)", -0.125, 0.020, 54 * MM, 24 * MM, 14 * MM, "#1c4fa3", "pcb")
        module("buck5", "5 V 3 A step-down converter (board logic)", -0.125, -0.012, 54 * MM, 24 * MM, 14 * MM, "#1c4fa3", "pcb")
        for q, (dx, dy) in enumerate(((-0.039, -0.0245), (0.019, -0.0245), (-0.039, 0.0245), (0.019, 0.0245))):
            add(cyl(f"pi.post{q}", "electronics", "M2.5 standoff, 6 mm", [-0.105 + dx, -0.095 + dy, zt0 + 6 * MM], [0, 0, 1], 2.5 * MM, 6 * MM, COL["brass"], mat="metal",
                    mount=mount("drawer.floor", "M2.5x8 screw from below")))
            self._screw(out, f"pi.post{q}.b", "M2.5", 8, [-0.105 + dx, -0.095 + dy, zt0 - tw], [0, 0, 1], through=["drawer.floor"], into=f"pi.post{q}", grip=tw / MM, group="electronics")
            self._screw(out, f"pi.post{q}.t", "M2.5", 5, [-0.105 + dx, -0.095 + dy, zt0 + 7.6 * MM], [0, 0, -1], through=["pi"], into=f"pi.post{q}", grip=1.6, group="electronics")
        module("pi", "host computer: Raspberry Pi 5 (85 x 56 mm)", -0.105, -0.095, 85 * MM, 56 * MM, 1.6 * MM, COL["pcb"], "pcb", lift=6 * MM, by="4x M2.5 standoff", to="pi.post0")
        module("pi.ports", "Raspberry Pi USB / Ethernet connectors", -0.071, -0.095, 17 * MM, 52 * MM, 15 * MM, COL["alu"], "metal", lift=7.6 * MM, by="soldered", to="pi")
        hx, hy = 0.030, -0.045
        for q, (dx, dy) in enumerate(((-0.052, -0.037), (0.052, -0.037), (-0.052, 0.037), (0.052, 0.037))):
            add(cyl(f"hub.post{q}", "electronics", "M3 standoff, 6 mm", [hx + dx, hy + dy, zt0 + 6 * MM], [0, 0, 1], 2.75 * MM, 6 * MM, COL["brass"], mat="metal",
                    mount=mount("drawer.floor", "M3x8 screw from below")))
            self._screw(out, f"hub.post{q}.b", "M3", 8, [hx + dx, hy + dy, zt0 - tw], [0, 0, 1], through=["drawer.floor"], into=f"hub.post{q}", grip=tw / MM, group="electronics")
            self._screw(out, f"hub.post{q}.t", "M3", 5, [hx + dx, hy + dy, zt0 + 7.6 * MM], [0, 0, -1], through=["hub"], into=f"hub.post{q}", grip=1.6, group="electronics")
        module("hub", "hub board: carries the modules below (110 x 80 mm prototyping board)", hx, hy, 110 * MM, 80 * MM, 1.6 * MM, COL["pcb"], "pcb", lift=6 * MM, by="4x M3 standoff", to="hub.post0")
        hz0 = 7.6 * MM

        def on_hub(pid, label, dx, dy, sx, sy, sz, color, mat="pcb"):
            module(pid, label, hx + dx, hy + dy, sx, sy, sz, color, mat, lift=hz0, by="soldered / pin header", to="hub")

        on_hub("pico", "bus hub: Raspberry Pi Pico (51 x 21 mm)", -0.025, 0.022, 51 * MM, 21 * MM, 3.5 * MM, COL["pcb"])
        on_hub("rs485", "RS-485 transceiver breakout (25 x 23 mm)", 0.030, 0.022, 25.4 * MM, 22.9 * MM, 3.0 * MM, COL["red"])
        for q in range(3):
            on_hub(f"ldd{q}", "UV string driver: Mean Well LDD-700L (22.6 x 9.9 x 8.9 mm)", -0.040 + q * 0.014, -0.022, 9.9 * MM, 22.6 * MM, 8.9 * MM, COL["black"], "plastic")
            on_hub(f"fet{q}", "strobe switch: MOSFET driver (25 x 18 mm)", 0.008 + q * 0.0195, -0.019, 17.7 * MM, 25.4 * MM, 7.2 * MM, "#111827")
        on_hub("valvedrv", "jet driver: H-bridge for the piezo pulse (adjustable +/-5 to 30 V) and the nozzle-heater switch", -0.005, 0.002, 40 * MM, 12 * MM, 8 * MM, "#0f5132")
        module("hbridge", "Peltier H-bridge: BTS7960 module (50 x 50 x 43 mm)", 0.125, -0.045, 50 * MM, 50 * MM, 43 * MM, COL["red"], "pcb")
        return out

    # ------------------------------------------------------------------ wiring
    def _wires(self, G, brd, feet, cams, lights, boards):
        """Every cable as a 3-D path.  Board-to-board jumpers and device leads run over the outside of
        the frame, clear of the ports; each of the three sectors has one bundle that drops to a hole in
        the floor next to a foot of the stand and continues to the hub in the drawer."""
        arr, rs, rt, fo = G["arr"], G["rs"], G["rt"], G["fo"]
        E = self._enc_dims()
        a, b, zt, sh = E["a"], E["b"], E["zt"], E["sh"]
        z_foam, z_sf, z_tray, zu = -E["zb"], E["z_sf"], E["z_tray"], E["z_unit"]
        R_sq = rs + self.can_h + self.pcb_t + self.back_h + 7 * MM
        R_tri = rt * fo + 7 * MM
        normals = [(f["n"], R_sq) for f in G["sq"]] + [(t["d"], R_tri) for t in [G["ports"]["top"], G["ports"]["bottom"]] + G["ports"]["upper"] + G["ports"]["lower"]]
        keep = [(G["ports"]["top"]["d"], np.radians(31), "top"), (G["ports"]["bottom"]["d"], np.radians(34), "bottom")]
        keep += [(t["d"], np.radians(25.5), f"u{q}") for q, t in enumerate(G["ports"]["upper"])]
        keep += [(t["d"], np.radians(25.5), f"l{q}") for q, t in enumerate(G["ports"]["lower"])]
        r_high = lambda own: 0.114 if own and own[0] == "u" else 0.099      # clears the camera stack / the low strobe and UV stars
        tri_of = {f"u{q}": t for q, t in enumerate(G["ports"]["upper"])}
        tri_of.update({f"l{q}": t for q, t in enumerate(G["ports"]["lower"])})

        def env(dv, own=None):
            r = min(R / float(n @ dv) for n, R in normals if n @ dv > 0.25)
            if own is not None:
                for pd, ang, nm in keep:
                    if nm == own and np.arccos(np.clip(pd @ dv, -1, 1)) < ang:
                        r = max(r, r_high(own))
            return r

        def clear(dv, own):
            return all(nm == own or np.arccos(np.clip(pd @ dv, -1, 1)) >= ka for pd, ka, nm in keep)

        def over(d0, d1, own=None, n_min=6):
            """Path over the frame from direction d0 to d1 that stays out of every port except `own`."""
            ang = np.arccos(np.clip(d0 @ d1, -1, 1))
            N = max(n_min, int(np.degrees(ang) / 5) + 2)
            ds = [unit(np.sin((1 - i / N) * ang) * d0 + np.sin(i / N * ang) * d1) if ang > 1e-6 else d0.copy() for i in range(N + 1)]
            for _ in range(60):
                for i in range(1, N):
                    for pd, ka, nm in keep:
                        if nm == own:
                            continue
                        c = float(np.clip(pd @ ds[i], -1, 1))
                        if np.arccos(c) < ka:
                            tang = ds[i] - c * pd
                            tang = unit(tang) if np.linalg.norm(tang) > 1e-6 else perp(pd)
                            ds[i] = unit(np.cos(ka + 0.01) * pd + np.sin(ka + 0.01) * tang)
                for i in range(1, N):
                    cand = unit(0.25 * ds[i - 1] + 0.5 * ds[i] + 0.25 * ds[i + 1])
                    if clear(cand, own) or not clear(ds[i], own):
                        ds[i] = cand
            return [dv * env(dv, own) for dv in ds]

        holes = self._pass_throughs(feet)

        def sector_of(p):
            az = np.arctan2(p[1], p[0])
            return int(np.argmin([abs((az - np.arctan2(h["xy"][1], h["xy"][0]) + np.pi) % (2 * np.pi) - np.pi) for h in holes[:3]]))

        def drop(k, lane=0.0):
            """From the lower edge of the frame straight down, then out to floor hole k and into the drawer."""
            hx, hy = holes[k]["xy"]
            az = np.arctan2(hy, hx)
            d_edge = unit([np.cos(az) * np.sin(np.radians(118)), np.sin(az) * np.sin(np.radians(118)), np.cos(np.radians(118))])
            p_edge = d_edge * env(d_edge)
            off = np.array([-np.sin(az), np.cos(az), 0.0]) * lane
            return d_edge, [p_edge + off, np.array([p_edge[0], p_edge[1], z_foam + 36 * MM]) + off, np.array([hx, hy, z_foam + 9 * MM]) + off * 0.25,
                            np.array([hx, hy, z_foam]) + off * 0.25, np.array([hx, hy, z_sf - 12 * MM]) + off * 0.25]

        def to_drawer(p_end, target, z_run=None):
            z_run = z_tray + 50 * MM if z_run is None else z_run
            target = np.asarray(target, float)
            return [np.array([p_end[0], p_end[1], z_run]), np.array([target[0], target[1], z_run]), target]

        W = []

        def cable(label, a_id, b_id, path, conductors, spacing=1.1 * MM, r=0.5 * MM):
            path = [np.asarray(p, float) for p in path]
            n = len(conductors)
            for i, fn in enumerate(conductors):
                o = (i - (n - 1) / 2) * spacing
                pts = []
                for j, p in enumerate(path):
                    tg = path[min(j + 1, len(path) - 1)] - path[max(j - 1, 0)]
                    tg = unit(tg) if np.linalg.norm(tg) > 1e-9 else np.array([0, 0, 1.0])
                    rad = unit(p) if np.linalg.norm(p) > 1e-4 else np.array([1.0, 0, 0])
                    bn = np.cross(tg, rad)
                    bn = unit(bn) if np.linalg.norm(bn) > 1e-3 else perp(tg)
                    pts.append(_L(p + o * bn))
                W.append(dict(label=label, fn=fn, color=WIRE[fn][0], r=float(r), a=a_id, b=b_id, pts=pts))

        HUB = np.array([0.030, -0.045, z_tray + 7.6 * MM + 2 * MM])
        hub = lambda dx, dy: HUB + np.array([dx, dy, 0.0])

        # --- board chains: three sectors of four boards (bottom to top), and one chain round the six boards of the lift-off cap.
        #     Only the cap chain's two trunk cables cross the cap joint; they unplug at the first cap board.
        conn = {i["ref"]: i for i in brd.items if i["ref"] in ("J1", "J2", "J3", "J4")}
        chains = [[], [], [], []]
        for bd_ in boards:
            chains[3 if bd_.get("cap") else sector_of(np.array(bd_["c"]))].append(bd_)
        for k in range(3):
            chains[k].sort(key=lambda q: q["c"][2])
        chains[3].sort(key=lambda q: (np.arctan2(q["c"][1], q["c"][0]) - np.pi / 2) % (2 * np.pi))       # start at the back
        k_cap = int(np.argmin([abs((np.arctan2(h["xy"][1], h["xy"][0]) - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) for h in holes[:3]]))

        def jpos(bd_, ref, h):
            it = conn[ref]
            return np.array(bd_["c"]) + it["x"] * MM * np.array(bd_["u"]) + it["y"] * MM * np.array(bd_["v"]) + h * np.array(bd_["n"])

        BUS = ["5v", "gnd", "rs485a", "rs485b", "sync", "gnd"]
        PWR = ["drive+", "gnd"]
        for k in range(4):
            ch = chains[k]
            kd = k if k < 3 else k_cap
            for kind, rin, rout, cond, h, lane in (("bus", "J1", "J2", BUS, 2.9 * MM, -7 * MM if k < 3 else -28 * MM), ("power", "J3", "J4", PWR, 5.0 * MM, 5 * MM if k < 3 else 28 * MM)):
                def lead(bd_, ref):
                    p = jpos(bd_, ref, h)
                    nn, vv = np.array(bd_["n"]), np.array(bd_["v"])
                    ex = vv * np.sign(conn[ref]["y"]) if kind == "bus" else nn          # bus plugs in from the edge, power from the top
                    p1 = p + ex * 5 * MM + nn * (2 * MM if kind == "bus" else 4 * MM)
                    return p, p1, unit(p1 + nn * 6 * MM)
                p, p1, dv = lead(ch[0], rin)
                d_edge, dr = drop(kd, lane)
                tgt = hub(-0.045 + 0.03 * k, 0.036) if kind == "bus" else np.array([-0.095, 0.020 - 0.006 * (k - 1), z_tray + 12 * MM])
                cable(f"{kind} trunk, sector {k+1}" if k < 3 else f"{kind} trunk, cap (unplugs at the first cap board)", f"{ch[0]['name']}.parts", "hub" if kind == "bus" else "buck18", [p, p1] + over(dv, d_edge)[:-1] + dr + to_drawer(dr[-1], tgt), cond,
                      r=0.45 * MM if kind == "bus" else 0.65 * MM)
                for a_, b_ in zip(ch[:-1], ch[1:]):
                    pa, pa1, da = lead(a_, rout)
                    pb, pb1, db = lead(b_, rin)
                    cable(f"{kind} jumper {a_['name']} -> {b_['name']}", f"{a_['name']}.parts", f"{b_['name']}.parts", [pa, pa1] + over(da, db) + [pb1, pb], cond,
                          r=0.45 * MM if kind == "bus" else 0.65 * MM)

        # --- devices on the ports
        def from_port(p, own, k, lane, target, a_id, b_id, label, cond, r=0.5 * MM, spacing=1.1 * MM):
            p = np.asarray(p, float)
            lift = unit(p) * max(np.linalg.norm(p) + 8 * MM, r_high(own))
            d_edge, dr = drop(k, lane)
            cable(label, a_id, b_id, [p, lift] + over(unit(lift), d_edge, own=own)[1:-1] + dr + to_drawer(dr[-1], target), cond, spacing=spacing, r=r)

        z_pi = z_tray + 7.6 * MM + 8 * MM
        for q, c in enumerate(cams):
            k = sector_of(np.array(c["port"]))
            from_port(c["usb"], f"u{q}", k, 12 * MM, [-0.0605, -0.107 + 0.012 * q, z_pi], f"cam{q}.usb", "pi.ports", f"camera {q+1} USB", ["usb"], r=1.75 * MM)
            from_port(c["trig"], f"u{q}", k, 16 * MM, hub(-0.05, 0.012 - 0.006 * q), f"cam{q}.pcb", "hub", f"camera {q+1} trigger", ["trigger", "gnd"], r=0.4 * MM, spacing=0.9 * MM)
        port_of = lambda l: ("u" if l["id"].startswith("uv_u") else "l") + l["id"][4]
        by_sector = {0: [], 1: [], 2: []}
        for l in [l for l in lights if l["kind"] == "uv"]:
            by_sector[sector_of(np.array(l["pos"]))].append(l)
        for k in range(3):
            leds = sorted(by_sector[k], key=lambda l: -l["pos"][2])
            if not leds:
                continue
            pads = []
            for l in leds:
                ax = np.array(l["dir"])
                s_back = np.array(l["pos"]) - ax * (6.8 * MM + 1.6 * MM + 0.4 * MM)          # just behind the star
                sv = perp(ax)
                pads.append((l, s_back + sv * 7 * MM, s_back - sv * 7 * MM))
            from_port(pads[0][1], port_of(leds[0]), k, -12 * MM, hub(-0.040 + 0.014 * k, -0.034), f"{leds[0]['id']}.star", f"ldd{k}", f"UV string {k+1} (+)", ["uv+"])
            for (la, _, a_minus), (lb, b_plus, _) in zip(pads[:-1], pads[1:]):
                oa, ob = port_of(la), port_of(lb)
                if oa == ob:
                    tp = tri_of[oa]
                    way = tp["d"] * (rt * fo + (7 if oa[0] == "u" else 13) * MM) - tp["w"][0] * (29 if oa[0] == "u" else 11) * MM   # round the camera pod / over the strobe
                    path = [a_minus, unit(a_minus) * (np.linalg.norm(a_minus) + 5 * MM), way, unit(b_plus) * (np.linalg.norm(b_plus) + 5 * MM), b_plus]
                else:
                    da, db = unit(a_minus), unit(b_plus)
                    pa = da * max(np.linalg.norm(a_minus) + 8 * MM, r_high(oa))
                    pb = db * max(np.linalg.norm(b_plus) + 8 * MM, r_high(ob))
                    md = unit(da + db)
                    path = [a_minus, pa] + over(da, md, own=oa)[1:] + over(md, db, own=ob)[1:-1] + [pb, b_plus]
                cable(f"UV string {k+1} link {la['id']} -> {lb['id']}", f"{la['id']}.star", f"{lb['id']}.star", path, ["uv+"])
            from_port(pads[-1][2], port_of(leds[-1]), k, -16 * MM, hub(-0.040 + 0.014 * k, -0.010), f"{leds[-1]['id']}.star", f"ldd{k}", f"UV string {k+1} (-)", ["gnd"])
        for q, l in enumerate([l for l in lights if l["kind"] == "strobe"]):
            dd = -np.array(l["dir"])
            p = np.array(l["pos"]) + dd * (0.5 * MM + 5.6 * MM + 0.4 * MM) + perp(dd) * 7 * MM
            from_port(p, f"l{q}", sector_of(dd), 20 * MM, hub(0.008 + 0.0195 * q, -0.036), f"strobe{q}.star", f"fet{q}", f"strobe {q+1}", ["strobe+", "gnd"])
        las = [l for l in lights if l["kind"] == "laser"][0]
        p = np.array(las["pos"]) - np.array(las["dir"]) * 33.4 * MM
        from_port(p, "u0", sector_of(p), 24 * MM, hub(0.05, 0.0), "laser", "hub", "line laser", ["laser", "gnd"], r=0.35 * MM, spacing=0.8 * MM)
        # air sensors (I2C)
        I2C = ["3v3", "gnd", "sda", "scl"]
        q_s = int(np.argmax([tt_["d"][1] for tt_ in G["ports"]["upper"]]))
        t1 = G["ports"]["upper"][q_s]
        d = t1["d"]
        pa = d * (rt + 0.5 * MM + 22 * MM + 2.5 * MM + 5.0 * MM) - t1["w"][0] * 48 * MM
        from_port(pa, f"u{q_s}", sector_of(d), -20 * MM, hub(0.05, 0.012), "bme_a", "hub", "air sensor (top)", I2C, r=0.3 * MM, spacing=0.7 * MM)
        f0 = feet[0]
        mid = 0.55 * f0["A"] + 0.45 * f0["B"]
        pb = mid - np.cross([0, 0, 1.0], f0["e"]) * 14 * MM + np.array([0, 0, 7 * MM]) + f0["e"] * 9 * MM
        k0 = int(np.argmin([np.hypot(h["xy"][0] - f0["xy"][0], h["xy"][1] - f0["xy"][1]) for h in holes[:3]]))
        hx0, hy0 = holes[k0]["xy"]
        path = [pb, pb + np.array([0, 0, 5 * MM]) + f0["e"] * 12 * MM, np.array([hx0, hy0, z_foam + 30 * MM]), np.array([hx0, hy0, z_foam]), np.array([hx0, hy0, z_sf - 12 * MM])]
        cable("air sensor (bottom)", "bme_b", "hub", path + to_drawer(path[-1], hub(0.05, 0.018)), I2C, r=0.3 * MM, spacing=0.7 * MM)
        # droplet generator: piezo pulse, nozzle heater + thermistor, light gate; all leave through the service hole in the floor
        sx, sy = holes[3]["xy"]
        zb_o = -rt * fo
        z_dock, z_ch = zb_o - 10 * MM, zb_o - 27.5 * MM
        svc = lambda off: [np.array([sx + off, sy, z_foam + 10 * MM]), np.array([sx + off, sy, z_foam]), np.array([sx + off, sy, z_sf - 12 * MM])]
        pz = np.array([0.0, 24.5 * MM, z_ch - 17 * MM])
        path = [pz, pz + np.array([0, 5 * MM, -6 * MM])] + svc(-2.0 * MM)
        cable("piezo pulse", "jet.clamp", "valvedrv", path + to_drawer(path[-1], hub(-0.005, 0.006)), ["valve", "valve"], r=0.35 * MM, spacing=0.8 * MM)
        ph = np.array([0.0, 11.5 * MM, self.nozzle_tip_z - 11 * MM])
        path = [ph, np.array([-1.2 * MM, 16.5 * MM, z_dock - 2 * MM]), np.array([-3.5 * MM, 17.5 * MM, z_dock - 8 * MM]), np.array([-4 * MM, 27 * MM, z_ch - 3 * MM])] + svc(-4.8 * MM)
        cable("nozzle heater and thermistor", "jet.block", "valvedrv", path + to_drawer(path[-1], hub(0.010, 0.006)), ["heater", "gnd", "therm", "gnd"], r=0.3 * MM, spacing=0.7 * MM)
        zg = self.nozzle_tip_z + 4 * MM
        path = [np.array([11.0 * MM, 6.75 * MM, zg]), np.array([13.3 * MM, 5 * MM, zb_o + 2 * MM]), np.array([13.6 * MM, 5.2 * MM, zb_o - 1.0 * MM]), np.array([17 * MM, 6 * MM, zb_o - 1.5 * MM]),
                np.array([25 * MM, 12 * MM, zb_o - 1.5 * MM]), np.array([22 * MM, 36 * MM, zb_o - 22 * MM]), np.array([8 * MM, 36 * MM, z_ch - 22 * MM])] + svc(2.5 * MM)
        cable("light gate", "gate.base", "hub", path + to_drawer(path[-1], hub(0.045, -0.006)), ["3v3", "gnd", "gate", "gnd"], r=0.3 * MM, spacing=0.65 * MM)
        f2 = feet[2]
        t2 = np.cross([0, 0, 1.0], f2["e"])
        rb = f2["e"] * 62 * MM + t2 * 24 * MM + np.array([0, 0, zb_o - 58 * MM])           # bottom end of the reservoir barrel
        path = [np.array([0, 30 * MM, z_ch - 9 * MM]), np.array([0, 37 * MM, z_ch - 13 * MM]) + t2 * 6 * MM, f2["e"] * 54 * MM + t2 * 18 * MM + np.array([0, 0, zb_o - 57 * MM]), rb]
        cable("resin feed", "jet.barb", "res.barrel", path, ["resin"], r=0.8 * MM)
        # interlocks: a loop from the hub through both switches (in the foam pocket), down the front-right corner
        ps = np.array([b - 7.2 * MM, -(a - 9.9 * MM), zt - 8.9 * MM])
        kk = int(np.argmin([np.hypot(h["xy"][0] - a, h["xy"][1] + a) for h in holes[:3]]))
        hk = np.array(holes[kk]["xy"])
        corner = [np.array([a - 3 * MM, -(a - 9.9 * MM), zt - 8 * MM]), np.array([a - 3 * MM, -(a - 3 * MM), zt - 26 * MM]), np.array([a - 3 * MM, -(a - 3 * MM), z_foam + 6 * MM]),
                  np.append(hk * 1.17, z_foam + 5 * MM), np.append(hk, z_foam), np.append(hk, z_sf - 12 * MM)]
        cable("interlock loop", "sw.door", "hub", [ps] + corner + to_drawer(corner[-1], hub(0.05, -0.02)), ["interlock", "interlock"], r=0.35 * MM, spacing=0.8 * MM)
        cable("interlock link", "sw.door", "sw.lid", [np.array([b - 7.2 * MM, -(a - 17 * MM), zt - 8.9 * MM]), np.array([b - 9 * MM, -(a - 21 * MM), zt - 7 * MM]), np.array([b - 7.2 * MM, -(a - 25 * MM), zt - 5.1 * MM])],
              ["interlock"], r=0.35 * MM)
        # thermal: the outside parts are wired down the outside of the wall and in through a hole in the drawer wall;
        # the inside fan lead runs down the wall and along the floor to the nearest floor hole
        yu = E["y_unit"]
        for sxn, nm in ((1, "right"), (-1, "left")):
            xo = sxn * (b + sh)
            for yy, label, cond, r_, sp, a_id, b_id, tgt in ((yu + 30 * MM, f"Peltier {nm}", ["tec+", "gnd"], 0.9 * MM, 2.0 * MM, f"tec_{nm}.sink_out", "hbridge", [0.125, -0.045 + 0.004 * sxn, z_tray + 44 * MM]),
                                                             (yu - 30 * MM, f"outside fan {nm}", ["fan", "gnd"], 0.4 * MM, 0.9 * MM, f"tec_{nm}.sink_out", "psu12", [0.075 + 0.01 * sxn, 0.0495, z_tray + 20 * MM])):
                path = [np.array([xo + sxn * 14 * MM, yy, zu - 46 * MM]), np.array([xo + sxn * 10 * MM, yy, z_tray + 36 * MM]), np.array([xo - sxn * 16 * MM, yy, z_tray + 36 * MM])]
                cable(label, a_id, b_id, path + to_drawer(path[-1], tgt, z_run=z_tray + 52 * MM), cond, r=r_, spacing=sp)
            kf = int(np.argmin([np.hypot(h["xy"][0] - sxn * a, h["xy"][1] - yu - 64 * MM) for h in holes[:3]]))
            hf = np.array(holes[kf]["xy"])
            xf = xo - sxn * 49.5 * MM
            path = [np.array([xf, yu + 61 * MM, zu - 61 * MM]), np.array([sxn * (a - 4 * MM), yu + 66 * MM, zu - 72 * MM]), np.array([sxn * (a - 4 * MM), yu + 66 * MM, z_foam + 6 * MM]),
                    np.append(hf * 1.17, z_foam + 5 * MM), np.append(hf, z_foam), np.append(hf, z_sf - 12 * MM)]
            cable(f"inside fan {nm}", f"tec_{nm}.fan", "hub", path + to_drawer(path[-1], hub(0.05, -0.03 + 0.008 * sxn)), ["fan", "gnd", "fan"], r=0.35 * MM, spacing=0.8 * MM)
        # drawer: supplies to the converters, the hub and the H-bridge
        zr = z_tray + 40 * MM
        t24 = lambda dx: np.array([-0.085 + dx, 0.0495, z_tray + 20 * MM])
        cable("24 V to drive converter", "psu24", "buck18", [t24(-0.03), t24(-0.03) + np.array([0, -0.01, 0.01]), np.array([-0.125, 0.034, z_tray + 10 * MM])], ["drive+", "gnd"], r=0.8 * MM, spacing=1.8 * MM)
        cable("24 V to 5 V converter", "psu24", "buck5", [t24(-0.055), np.array([-0.156, 0.03, zr]), np.array([-0.156, -0.012, z_tray + 10 * MM])], ["drive+", "gnd"], r=0.6 * MM, spacing=1.4 * MM)
        cable("24 V to UV drivers", "psu24", "hub", [t24(0.02), np.array([-0.04, 0.02, zr]), hub(-0.05, -0.03)], ["drive+", "gnd"], r=0.6 * MM, spacing=1.4 * MM)
        cable("5 V to hub", "buck5", "hub", [np.array([-0.096, -0.012, z_tray + 10 * MM]), np.array([-0.06, -0.03, zr - 14 * MM]), hub(-0.055, -0.02)], ["5v", "gnd"], r=0.6 * MM, spacing=1.4 * MM)
        cable("5 V to host", "buck5", "pi", [np.array([-0.125, -0.0255, z_tray + 10 * MM]), np.array([-0.13, -0.05, zr - 14 * MM]), np.array([-0.13, -0.068, z_tray + 9.5 * MM])], ["5v", "gnd"], r=0.6 * MM, spacing=1.4 * MM)
        cable("12 V to H-bridge", "psu12", "hbridge", [np.array([0.11, 0.0495, z_tray + 20 * MM]), np.array([0.135, 0.0, zr + 10 * MM]), np.array([0.135, -0.045, z_tray + 44 * MM])], ["tec+", "gnd"], r=0.9 * MM, spacing=2.0 * MM)
        cable("hub to host (USB)", "pico", "pi.ports", [np.array([0.030 - 0.025 - 0.0265, -0.045 + 0.022, z_tray + 10 * MM]), np.array([-0.04, -0.06, zr - 14 * MM]), np.array([-0.0605, -0.083, z_pi])], ["usb"], r=1.75 * MM)
        return W

    # ------------------------------------------------------------------ sourcing / cost
    def bom(self):
        """(item, qty, unit USD, where, status).  status: 'verified' = a supplier page or the supplier's parts library was read
        for this exact part on 2026-11-10; 'listed' = seen in a listing without a live price; 'generic' = commodity item."""
        n = self.array().n
        if not getattr(self, "_fasteners", None):
            self.build()
        fz = self.fasteners()
        cnt = lambda key: sum(v for k, v in fz.items() if key in k)
        return [
            ("40 kHz transmitter, 10 mm can, Sunny TCT40-10T2, guaranteed 110 dB (+25 spares)", n + 25, 0.24, "Maritex", "verified"),
            ("TB-25 board: 4-layer PCB, parts and back-side assembly at JLCPCB (order 20; parts 15.7 USD each from their library prices, board and fees estimated)", 20, 21.0, "JLCPCB", "verified"),
            ("Raspberry Pi Debug Probe (programs the boards through the 3-pin SWD connector)", 1, 12.0, "Raspberry Pi resellers", "listed"),
            ("JST-SH 6-pin cable, 150 mm (bus jumpers and trunks)", 18, 3.32, "Pololu 5543", "verified"),
            ("JST-PH 2-pin cable (power jumpers and trunks)", 18, 0.75, "Adafruit 261", "verified"),
            (f"Screws, by count in the model: {cnt('M2 x')} M2, {cnt('M2.5 x')} M2.5, {cnt('M3 x') - cnt('thumb')} M3, {cnt('M4 x')} M4, {cnt('M5 x')} M5, {cnt('wood')} wood screws "
             "(assortment boxes M2 / M2.5 / M3 / M4, see hardware/fasteners.json)", 1, 24.0, "any", "generic"),
            (f"M3 knurled thumb screws ({cnt('thumb')} in the model: pin clamp, chuck, cap, cartridge lock)", 1, 8.0, "any", "generic"),
            (f"Heat-set inserts ({cnt('insert')} in the model: M2.5, M3, M4, M5 assortment)", 1, 21.0, "CNC Kitchen / Prusa", "verified"),
            ("PETG filament for frame, legs, dock, pods, chuck, cartridge body (~0.7 kg used)", 1, 24.0, "Prusament", "verified"),
            ("Global-shutter camera InnoMaker U20CAM-9281M with trigger input and 2.8 mm M12 lens", 3, 32.99, "inno-maker.com", "verified"),
            ("405 nm LED Luminus SST-10-UV-A130-G405 + 20 mm star board for 3535", 11, 4.2, "Digi-Key / Cutter", "verified"),
            ("Lens for the UV LEDs, Ledil LISA2-RS-PIN (10 mm)", 11, 1.68, "Digi-Key", "verified"),
            ("UV string driver Mean Well LDD-700L", 3, 3.25, "Digi-Key", "verified"),
            ("Red LED on star, Cree XP-E2", 3, 5.25, "LEDSupply", "verified"),
            ("Opal acrylic sheet for strobe diffusers", 1, 10.7, "US Plastic", "listed"),
            ("MOSFET driver for the strobes", 3, 3.95, "Adafruit 5648", "verified"),
            ("Line laser 650 nm 5 mW", 1, 8.95, "Adafruit 1057", "verified"),
            ("Slotted photo-interrupter Vishay TCST2300 (droplet light gate)", 2, 1.54, "Digi-Key", "verified"),
            ("BME280 temperature / humidity / pressure breakout", 2, 14.95, "Adafruit 2652", "verified"),
            ("Raspberry Pi Pico (bus hub) + RS-485 breakout + prototyping board", 1, 20.5, "Raspberry Pi / SparkFun", "verified"),
            ("Drive supply Mean Well LRS-100-24 (24 V 4.5 A, trims to 27 V)", 1, 17.6, "Mouser", "listed"),
            ("Thermal supply 12 V 12.5 A (LRS-150-12 class)", 1, 22.0, "Mouser", "generic"),
            ("Step-down converters: adjustable 5 A (drive voltage) and 5 V 3 A", 2, 5.0, "any", "generic"),
            ("Peltier assembly 12 V 5 A with heat sink and fan", 2, 34.95, "Adafruit 1335", "verified"),
            ("Inside heat sink 100 x 70 x 25 mm + 120 mm PWM fan (Arctic P12)", 2, 16.0, "any / Arctic", "listed"),
            ("H-bridge module BTS7960", 1, 5.0, "AliExpress", "listed"),
            ("Melamine foam sheet 48 x 24 x 1 in", 1, 49.12, "Penn Tool Co", "verified"),
            ("Plywood 1/4 in for enclosure and drawer (4 panels 24 x 24 in), wood glue, glue blocks", 4, 15.0, "any", "generic"),
            ("405 nm blocking acrylic 24 x 12 x 1/8 in (door window)", 1, 34.99, "J Tech Photonics", "verified"),
            ("Interlock switch Omron SS-5GL", 2, 2.09, "Digi-Key", "verified"),
            ("Hinges (4), magnetic catches (6), gasket tape, cable ties and clips, wire", 1, 45.0, "any", "generic"),
            ("Build pins: 2.5 mm stainless rod, cut to 100 mm (20 pcs)", 1, 8.0, "any", "generic"),
            ("UV resin 1 kg (thinnest found: Anycubic High Speed 2.0, 75-85 mPa.s)", 1, 32.0, "Anycubic", "listed"),
            ("Droplet generator: piezo disc CUI CEB-35D26 (2), spare included", 2, 2.5, "Digi-Key 102-1128-ND", "listed"),
            ("Droplet generator: brass M6 3-D printer nozzles 0.2 / 0.25 / 0.3 / 0.4 mm (set)", 1, 9.0, "any 3-D printer parts shop", "generic"),
            ("Droplet generator: hot-end heater block + M6 heat break + 24 V heater cartridge + 100 k thermistor", 1, 12.0, "any 3-D printer parts shop", "generic"),
            ("Droplet generator: O-rings (1-1/4 in ID x 1/16 in for the piezo; 1/8 in ID for the feed tube), 10-24 hose barbs, 1/8 in opaque tubing, 10 mL syringes", 1, 22.0, "McMaster-Carr 9452K113, 94115K006", "listed"),
            ("Droplet generator: dual H-bridge for the piezo pulse (L293D or DRV8871) + small adjustable 5-30 V converter + logic-level MOSFET for the heater", 1, 14.0, "Digi-Key / any", "generic"),
        ]

    def to_json(self, path, mesh_path=None, asm=None):
        """Scene for the viewer.  Printed solids are tessellated into `mesh_path`."""
        asm = asm or self.build()
        arr = self.array()
        bom = self.bom()
        parts = []
        blob = bytearray()
        for p in asm["parts"]:
            q = {k: v for k, v in p.items() if k != "ops"}
            if p["kind"] == "solid" or p.get("holes") or p.get("cuts") or p.get("bore", 0) > 0:     # anything that is not a plain box / cylinder
                s = _solid(p)
                vs, tris = s.tessellate(0.08, 0.35)
                v = np.array([[x.X, x.Y, x.Z] for x in vs], np.float32) * MM
                t = np.array(tris, np.uint32)
                q["mesh"] = dict(voff=len(blob), nv=int(len(v)), ioff=len(blob) + v.nbytes, ni=int(t.size))
                blob += v.tobytes() + t.tobytes()
                for kdrop in ("v_in", "v_out", "base", "vec"):
                    q.pop(kdrop, None)
            parts.append(q)
        if mesh_path:
            open(mesh_path, "wb").write(bytes(blob))
        data = dict(name=self.name, half=self.half, params=asdict(self), n_transducers=arr.n, components=parts, wires=asm["wires"],
                    cameras=asm["cameras"], lights=asm["lights"], boards=asm["boards"], wire_legend={k: dict(color=c, meaning=m) for k, (c, m) in WIRE.items()},
                    enclosure=self._enc_dims(),
                    fasteners=self.fasteners(),
                    bom=[dict(item=i, qty=q, unit_usd=u, total_usd=round(q * u, 2), where=w, status=s) for i, q, u, w, s in bom],
                    bom_total_usd=round(sum(q * u for _, q, u, _, _ in bom), 2))
        json.dump(data, open(path, "w"))
        return data


# ----------------------------------------------------------------------------
# solids, STEP / STL export, checks
# ----------------------------------------------------------------------------
def _cylinder(p, axis, r, h):
    """Cylinder from point p along +axis (mm)."""
    import build123d as bd
    return bd.Cylinder(r, h, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN)).moved(bd.Plane(origin=tuple(p), z_dir=tuple(axis)).location)


def _solid(c):
    import build123d as bd
    V = bd.Vector
    k = 1.0 / MM
    wire = lambda pts: bd.Wire.make_polygon([V(*(np.array(p) * k)) for p in pts], close=True)
    if c["kind"] == "cyl":
        s = bd.Cylinder(c["r"] * k, c["h"] * k, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
        if c.get("bore", 0) > 0:
            s = s - bd.Cylinder(c["bore"] * k, c["h"] * k + 2, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX)).moved(bd.Location((0, 0, 1)))
        s = s.moved(bd.Plane(origin=tuple(np.array(c["top"]) * k), z_dir=tuple(c["axis"])).location)
    elif c["kind"] == "box":
        sx, sy, sz = (np.array(c["size"]) * k).tolist()
        s = bd.Box(sx, sy, sz).moved(bd.Plane(origin=tuple(np.array(c["center"]) * k), x_dir=tuple(c["x"]), z_dir=tuple(c["z"])).location)
    elif c["shape"] == "frustum":
        s = bd.Solid.make_loft([wire(c["v_in"]), wire(c["v_out"])], True)
    else:
        s = bd.Solid.extrude(bd.Face(wire(c["base"])), V(*(np.array(c["vec"]) * k)))
    for h in c.get("holes", []):
        s = s - bd.Cylinder(h["r"] * k, 800.0).moved(bd.Plane(origin=tuple(np.array(h["p"]) * k), z_dir=tuple(h["axis"])).location)
    for q in c.get("cuts", []):
        sx, sy, sz = (np.array(q["size"]) * k).tolist()
        s = s - bd.Box(sx, sy, sz).moved(bd.Location(tuple(np.array(q["center"]) * k)))
    for op in c.get("ops", []):
        if op["op"] == "cyl":
            s = s - _cylinder(np.array(op["p"]) * k, op["axis"], op["r"] * k, op["h"] * k)
        elif op["op"] == "prism":
            s = s - bd.Solid.extrude(bd.Face(wire(op["base"])), V(*(np.array(op["vec"]) * k)))
        elif op["op"] == "cone":
            ax = unit(op["axis"]); u_ = perp(ax); v_ = np.cross(ax, u_)
            circ = lambda cz, r: [np.array(op["p"]) * k + ax * cz + r * k * (np.cos(2 * np.pi * i / 48) * u_ + np.sin(2 * np.pi * i / 48) * v_) for i in range(48)]
            s = s - bd.Solid.make_loft([bd.Wire.make_polygon([V(*q) for q in circ(0.0, op["r0"])], close=True),
                                        bd.Wire.make_polygon([V(*q) for q in circ(op["h"] * k, op["r1"])], close=True)], True)
        elif op["op"] == "half":
            s = bd.split(s, bisect_by=bd.Plane.XY, keep=bd.Keep.TOP if op["keep"] == "top" else bd.Keep.BOTTOM)
    sol = s.solids()
    if len(sol) == 1:
        return sol[0]
    return max(sol, key=lambda x: x.volume) if c.get("largest") else s


def build_solids(machine: Machine, asm=None):
    asm = asm or machine.build()
    return asm["parts"], [_solid(c) for c in asm["parts"]]


def _bbox(solids):
    lo = np.array([[b.min.X, b.min.Y, b.min.Z] for b in (s.bounding_box() for s in solids)])
    hi = np.array([[b.max.X, b.max.Y, b.max.Z] for b in (s.bounding_box() for s in solids)])
    return lo, hi


def interference_report(parts, solids, tol_mm3=2e-3):
    """Exact solid-solid interference check.  Broad phase on bounding boxes, then a boolean
    intersection for every candidate pair.  Pieces of the same printed body are allowed to
    overlap (they are printed as one).  Returns (pairs_checked, clashes)."""
    lo, hi = _bbox(solids)
    eps = 1e-4
    clashes, checked = [], 0
    n = len(solids)
    for i in range(n):
        ov = np.all((lo[i] < hi[i + 1:] - eps) & (hi[i] > lo[i + 1:] + eps), axis=1)
        for j in np.nonzero(ov)[0] + i + 1:
            if parts[i].get("body") and parts[i].get("body") == parts[j].get("body"):
                continue
            checked += 1
            inter = solids[i] & solids[j]
            vol = 0.0 if inter is None else float(getattr(inter, "volume", 0.0) or 0.0)
            if vol > tol_mm3:
                clashes.append((parts[i]["id"], parts[j]["id"], vol))
    return checked, clashes


def attachment_report(parts, solids, tol_mm=0.3):
    """Nothing may float.  Checks, for every part:
      1. it names what it is fastened to (or belongs to a printed body that does);
      2. it actually touches that part (gap <= tol_mm);
      3. following the fastenings leads to the enclosure floor.
    and for every printed body: its pieces touch each other, so it prints as one piece.
    Returns dict(n_parts, unmounted, gaps, unrooted, split_bodies)."""
    ids = {p["id"]: i for i, p in enumerate(parts)}
    bodies = {}
    for i, p in enumerate(parts):
        if p.get("body"):
            bodies.setdefault(p["body"], []).append(i)
    lo, hi = _bbox(solids)

    def dist(i, j):
        gap = np.maximum(np.maximum(lo[i] - hi[j], lo[j] - hi[i]), 0)
        if np.linalg.norm(gap) > tol_mm + 1e-9:
            return float(np.linalg.norm(gap))                 # bounding boxes are already further apart
        return float(solids[i].distance_to(solids[j]))

    unmounted, gaps, parent = [], [], {}
    for i, p in enumerate(parts):
        if p.get("root"):
            parent[i] = None
            continue
        m = p.get("mount")
        if not m:
            unmounted.append(p["id"]); continue
        tgt = [ids[m["to"]]] if m["to"] in ids else bodies.get(m["to"], [])
        if not tgt:
            unmounted.append(p["id"] + " -> unknown " + m["to"]); continue
        if p.get("body"):
            members = bodies[p["body"]]
            dmin, jbest = min(((dist(a, j), j) for a in members for j in tgt), key=lambda x: x[0])
        else:
            dmin, jbest = min(((dist(i, j), j) for j in tgt), key=lambda x: x[0])
        if dmin > tol_mm:
            gaps.append((p["id"], m["to"], round(dmin, 3)))
        parent[i] = jbest
    unrooted = []
    for i, p in enumerate(parts):
        seen, j = set(), i
        while j is not None and j in parent and j not in seen:
            seen.add(j); j = parent[j]
        if j is not None:
            unrooted.append(p["id"])
    split_bodies = []
    for name, members in bodies.items():
        comp = {m: m for m in members}

        def find(x):
            while comp[x] != x:
                comp[x] = comp[comp[x]]; x = comp[x]
            return x
        for a_i, a in enumerate(members):
            for b in members[a_i + 1:]:
                if find(a) != find(b) and dist(a, b) <= 0.05:
                    comp[find(a)] = find(b)
        if len({find(m) for m in members}) > 1:
            split_bodies.append((name, len({find(m) for m in members})))
    return dict(n_parts=len(parts), unmounted=unmounted, gaps=gaps, unrooted=unrooted, split_bodies=split_bodies,
                bodies={k: len(v) for k, v in bodies.items()})


def wire_report(parts, solids, wires, end_tol_mm=6.0, step_mm=2.5, skip_mm=9.0):
    """Every wire must start and end on its named parts and must not run through a solid.
    Returns dict(n_wires, length_m, dangling, through)."""
    import build123d as bd
    ids = {p["id"]: i for i, p in enumerate(parts)}
    lo, hi = _bbox(solids)
    dangling, through = [], {}
    total = 0.0
    for w in wires:
        pts = np.array(w["pts"]) / MM
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total += seg.sum()
        for end, pid in ((pts[0], w["a"]), (pts[-1], w["b"])):
            if pid not in ids:
                dangling.append((w["label"], w["fn"], "unknown part " + pid)); continue
            i = ids[pid]
            if np.any(end < lo[i] - end_tol_mm) or np.any(end > hi[i] + end_tol_mm):
                dangling.append((w["label"], w["fn"], f"{pid}: end is {float(np.linalg.norm(np.maximum(np.maximum(lo[i] - end, end - hi[i]), 0))):.1f} mm from the part"))
        s = np.concatenate([[0], np.cumsum(seg)])
        ts = np.arange(skip_mm, s[-1] - skip_mm, step_mm)
        if len(ts) == 0:
            continue
        samp = np.stack([np.interp(ts, s, pts[:, d]) for d in range(3)], 1)
        for q in samp:
            cand = np.nonzero(np.all((q > lo + 0.2) & (q < hi - 0.2), axis=1))[0]
            for i in cand:
                if parts[i].get("hidden"):
                    continue
                if solids[i].is_inside(bd.Vector(*q), 1e-4):
                    through.setdefault((w["label"], parts[i]["id"]), 0)
                    through[(w["label"], parts[i]["id"])] += 1
    return dict(n_wires=len(wires), length_m=round(total / 1000, 2), dangling=dangling, through=[(k[0], k[1], v) for k, v in through.items()])


def motion_report(parts, solids, motions, tol_mm3=0.05):
    """Things that are taken out or swung open must not hit anything on the way.
    motions: list of dict(name, moving=[indices], ignore=[indices], steps=[("move", (dx, dy, dz) mm) | ("turn", origin mm, axis, degrees)],
    extra=[solids in mm that travel with the moving parts]).  Returns one record per motion with the collisions found."""
    import build123d as bd
    lo, hi = _bbox(solids)
    out = []
    for mo in motions:
        mv = list(mo["moving"])
        skip = set(mv) | set(mo.get("ignore", []))
        static = np.array([i for i in range(len(parts)) if i not in skip])
        hits = {}
        n_pairs = 0
        for st in mo["steps"]:
            def tf(sol):
                return sol.moved(bd.Location(tuple(st[1]))) if st[0] == "move" else sol.rotate(bd.Axis(tuple(st[1]), tuple(st[2])), st[3])
            moved = [(parts[i]["id"], tf(solids[i])) for i in mv] + [(f"(travels along) {k}", tf(e)) for k, e in mo.get("extra", [])]
            for pid, ms in moved:
                bb = ms.bounding_box()
                l, h = np.array([bb.min.X, bb.min.Y, bb.min.Z]), np.array([bb.max.X, bb.max.Y, bb.max.Z])
                cand = static[np.all((lo[static] < h - 1e-4) & (hi[static] > l + 1e-4), axis=1)]
                for j in cand:
                    n_pairs += 1
                    inter = ms & solids[j]
                    vol = 0.0 if inter is None else float(getattr(inter, "volume", 0.0) or 0.0)
                    if vol > tol_mm3:
                        key = (pid, parts[j]["id"])
                        hits[key] = max(hits.get(key, 0.0), vol)
        out.append(dict(name=mo["name"], moving_parts=len(mv), steps=len(mo["steps"]), pairs_checked=n_pairs,
                        collisions=[dict(moving=a, hits=b, mm3=round(v, 2)) for (a, b), v in sorted(hits.items(), key=lambda kv: -kv[1])]))
    return out


def neck_report(stl_path, pitch=0.8, min_voxels=40):
    """Does a printed part hold together by more than a thread?  The STL is voxelised, one voxel layer is peeled off every
    surface (which removes anything thinner than about 2 mm), and the pieces that remain are counted.  More than one piece
    means that somewhere the part hangs together only by a thin neck or an edge contact."""
    import trimesh
    from scipy import ndimage as ndi
    m = trimesh.load(stl_path)
    if m.volume < 30000:                                   # small parts get a finer grid (peels 0.6 mm)
        pitch = 0.6
    vox = m.voxelized(pitch).fill().matrix
    core = ndi.binary_erosion(np.pad(vox, 2), structure=np.ones((3, 3, 3), bool))
    lab, n = ndi.label(core)
    sizes = np.bincount(lab.ravel())[1:]
    big = sizes[sizes >= min_voxels]
    return dict(voxels=int(vox.sum()), pieces_after_peeling=int(len(big)), largest_share=round(float(big.max() / max(big.sum(), 1)), 3) if len(big) else 0.0)


def export(machine: Machine, outdir, parts=None, solids=None):
    """STEP of the whole assembly, and one STL per printed body."""
    import build123d as bd
    if solids is None:
        parts, solids = build_solids(machine)
    os.makedirs(f"{outdir}/print", exist_ok=True)
    groups, bodies = {}, {}
    for c, s in zip(parts, solids):
        if c.get("hidden"):
            continue
        s.label = c["label"]; s.color = bd.Color(c["color"])
        if c.get("body"):
            bodies.setdefault(c["body"], []).append(s)
        else:
            groups.setdefault(c["group"], []).append(s)
    printed = {}
    for name, pieces in bodies.items():
        try:
            fused = pieces[0]
            for q in pieces[1:]:
                fused = fused + q
            sol = fused.solids()
            printed[name] = dict(solids=len(sol), volume_cm3=round(sum(x.volume for x in sol) / 1000, 1))
            fused.label = name
            body = fused
        except Exception as e:                                   # keep the pieces if the union fails
            body = bd.Compound(children=pieces, label=name)
            printed[name] = dict(solids=len(pieces), volume_cm3=round(sum(x.volume for x in pieces) / 1000, 1), note="union failed: " + repr(e)[:80])
        bd.export_stl(body, f"{outdir}/print/{name}.stl", tolerance=0.05, angular_tolerance=0.3)
        bb = body.bounding_box()
        printed[name]["size_mm"] = [round(bb.max.X - bb.min.X, 1), round(bb.max.Y - bb.min.Y, 1), round(bb.max.Z - bb.min.Z, 1)]
        groups.setdefault("printed", []).append(body)
    asm = bd.Compound(children=[bd.Compound(children=p, label=g) for g, p in groups.items()], label=machine.name)
    bd.export_step(asm, f"{outdir}/replicator_mk2.step")
    return printed
