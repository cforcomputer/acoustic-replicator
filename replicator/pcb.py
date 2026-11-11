"""Transducer board ("TB-25"): parts, footprints, placement, pin-level netlist.

One identical square board per square face of the frame.  Front: 5 x 5 40 kHz
transducers (through-hole, soldered by hand from the back).  Back: one RP2040 that
generates the 25 phased 40 kHz square waves on GPIO2..26 (one PIO program), 13 dual
30 V drivers, an ultrasonic MEMS microphone that listens through a hole between four
cans, RS-485, a programming connector, and the support passives.  All boards share
+V_DRIVE, +5 V, an RS-485 command bus and a hard SYNC line.

Every surface-mount footprint is the one JLCPCB's own parts library uses for the
part number in `LCSC` (downloaded once into hardware/pcb/lib/, see `fetch_lib`), so
pads, pin numbers and the pick-and-place origin agree with what their line places.

This module does placement and the netlist; `pcbroute` routes the copper, writes
the fabrication and assembly files and checks them.

Coordinates: millimetres, origin at the board centre, seen from the FRONT (the
transducer side).  Back-side parts are therefore mirrored in x.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = os.path.join(os.path.dirname(HERE), "hardware", "pcb", "lib")

TRANSDUCER = dict(name="40 kHz transmitter, 10 mm can (MA40S4S / MSO-P1040H07T class)", can_d=9.9, can_h=7.1, lead_pitch=5.0,
                  lead_d=0.64, drill=1.0, pad_d=1.8, cap_nF=2.55, v_max_pp=20.0)

# part type -> JLCPCB / LCSC number whose library footprint is used
LCSC = dict(RP2040="C2040", FLASH="C131025", XTAL="C20625731", LDO="C6186", DRV="C171845", RS485="C8963", MIC="C2879853",
            J_BUS="C160405", J_SWD="C160403", J_PWR="C160352", SOT23="C20526", C0402="C1525", R0402="C25744", C0603="C15849",
            R0603="C22787", LED0603="C2286", C0805="C15850", C1206="C13585")
# body (x, y, height) in the library orientation, colour, description
BODY = {
    "RP2040": ((7.0, 7.0, 0.9), "#1b1b1f", "QFN-56 7x7 mm, 0.4 mm pitch"), "FLASH": ((5.3, 5.3, 2.0), "#1b1b1f", "SOIC-8 208 mil"),
    "XTAL": ((3.2, 2.5, 0.8), "#c9ccd1", "3225 crystal"), "LDO": ((3.5, 6.5, 1.7), "#1b1b1f", "SOT-223"),
    "DRV": ((4.9, 3.9, 1.75), "#1b1b1f", "SOIC-8 150 mil"), "RS485": ((4.9, 3.9, 1.75), "#1b1b1f", "SOIC-8 150 mil"),
    "MIC": ((2.65, 3.5, 1.0), "#d4af37", "MEMS microphone 3.5 x 2.65 mm, bottom port"),
    "J_BUS": ((9.0, 4.25, 2.95), "#efe9d8", "JST-SH 6-pin side entry"), "J_SWD": ((6.0, 4.25, 2.95), "#efe9d8", "JST-SH 3-pin side entry"),
    "J_PWR": ((7.9, 5.0, 5.0), "#efe9d8", "JST-PH 2-pin top entry"), "SOT23": ((1.3, 2.9, 1.0), "#1b1b1f", "SOT-23"),
    "C0402": ((1.0, 0.5, 0.5), "#b08d57", "0402"), "R0402": ((1.0, 0.5, 0.35), "#26262b", "0402"), "C0603": ((1.6, 0.8, 0.8), "#b08d57", "0603"),
    "R0603": ((1.6, 0.8, 0.45), "#26262b", "0603"), "LED0603": ((1.6, 0.8, 0.6), "#e2483a", "0603 LED"),
    "C0805": ((2.0, 1.25, 1.25), "#b08d57", "0805"), "C1206": ((3.2, 1.6, 1.6), "#b08d57", "1206"),
}
# what is ordered for each value (JLCPCB library, read on 2026-11-10); basic = no extended-part fee
ORDER = {
    "RP2040": ("RP2040", "C2040", "extended"), "FLASH": ("W25Q16JVSSIQ, 16 Mbit QSPI flash", "C131025", "extended"),
    "XTAL": ("ABM8-272-T3, 12 MHz, CL 10 pF (the crystal Raspberry Pi specifies)", "C20625731", "extended"),
    "LDO": ("AMS1117-3.3", "C6186", "basic"), "DRV": ("IX4427NTR dual 1.5 A driver, 4.5-30 V", "C171845", "extended"),
    "RS485": ("SP3485EN-L/TR", "C8963", "basic"), "MIC": ("SPH0641LU4H-1 ultrasonic PDM microphone", "C2879853", "extended"),
    "J_BUS": ("SM06B-SRSS-TB", "C160405", "extended"), "J_SWD": ("SM03B-SRSS-TB", "C160403", "extended"), "J_PWR": ("B2B-PH-SM4-TB", "C160352", "extended"),
    "SOT23": ("MMBT3904", "C20526", "basic"), "100nF": ("100 nF 16 V X7R 0402", "C1525", "basic"), "1uF": ("1 uF 25 V X5R 0402", "C52923", "basic"),
    "15pF": ("15 pF 50 V C0G 0402", "C1548", "basic"), "1uF50": ("1 uF 50 V X5R 0603", "C15849", "basic"), "10uF": ("10 uF 25 V X5R 0805", "C15850", "basic"),
    "10uF50": ("10 uF 50 V X5R 1206", "C13585", "basic"), "10k": ("10 k 1 % 0402", "C25744", "basic"), "1k": ("1 k 1 % 0402", "C11702", "basic"),
    "100R": ("100 R 1 % 0402", "C25076", "basic"), "120R": ("120 R 1 % 0603 (not fitted by the assembler: bus termination, last board of a chain only)", "C22787", "basic"),
    "LED": ("red LED 0603", "C2286", "basic"),
}
PARTS = {"DRV": dict(v_max=30.0, outputs=2, i_peak=1.5), "RP2040": dict(gpio=30)}

# pin functions by pad number
PINS = {
    "RP2040": dict(zip(range(1, 58), ["IOVDD", "GPIO0", "GPIO1", "GPIO2", "GPIO3", "GPIO4", "GPIO5", "GPIO6", "GPIO7", "IOVDD", "GPIO8", "GPIO9", "GPIO10", "GPIO11",
               "GPIO12", "GPIO13", "GPIO14", "GPIO15", "TESTEN", "XIN", "XOUT", "IOVDD", "DVDD", "SWCLK", "SWDIO", "RUN", "GPIO16", "GPIO17",
               "GPIO18", "GPIO19", "GPIO20", "GPIO21", "IOVDD", "GPIO22", "GPIO23", "GPIO24", "GPIO25", "GPIO26", "GPIO27", "GPIO28", "GPIO29", "IOVDD",
               "ADC_AVDD", "VREG_VIN", "VREG_VOUT", "USB_DM", "USB_DP", "USB_VDD", "IOVDD", "DVDD", "QSPI_SD3", "QSPI_SCLK", "QSPI_SD0", "QSPI_SD2",
               "QSPI_SD1", "QSPI_SS", "GND"])),
    "FLASH": {1: "QSPI_SS", 2: "QSPI_SD1", 3: "QSPI_SD2", 4: "GND", 5: "QSPI_SD0", 6: "QSPI_SCLK", 7: "QSPI_SD3", 8: "+3V3"},
    "XTAL": {1: "XIN", 2: "GND", 3: "XTAL2", 4: "GND"},
    "LDO": {1: "GND", 2: "+3V3", 3: "+5V", 4: "+3V3"},
    "RS485": {1: "UART_RX", 2: "DE", 3: "DE", 4: "GND", 5: "GND", 6: "RS485_A", 7: "RS485_B", 8: "+3V3"},
    "MIC": {1: "PDM_DAT", 2: "GND", 3: "GND", 4: "PDM_CLK", 5: "+3V3"},
    "J_BUS": {1: "+5V", 2: "GND", 3: "RS485_A", 4: "RS485_B", 5: "SYNC_BUS", 6: "GND", 7: None, 8: None},
    "J_PWR": {1: "+VDRIVE", 2: "GND", 3: None, 4: None},
    "J_SWD": {1: "SWCLK", 2: "GND", 3: "SWDIO", 4: None, 5: None},
    "SOT23": {1: "QB", 2: "GND", 3: "DE"},
}
RP2040_NET = {"IOVDD": "+3V3", "USB_VDD": "+3V3", "ADC_AVDD": "+3V3", "VREG_VIN": "+3V3", "DVDD": "+1V1", "VREG_VOUT": "+1V1", "TESTEN": "GND", "GND": "GND",
              "GPIO0": "UART_TX", "GPIO1": "UART_RX", "GPIO2": "PDM_CLK", "GPIO3": "PDM_DAT", "GPIO29": "SYNC", "USB_DM": None, "USB_DP": None}
DRIVE_GPIO = [f"GPIO{i}" for i in range(4, 29)]            # 25 consecutive pins: one PIO "out pins, 25" with base 4


def fetch_lib(codes=None):
    """Download JLCPCB / EasyEDA library entries that are not on disk yet (read-only request, no account)."""
    import urllib.request
    os.makedirs(LIBDIR, exist_ok=True)
    for c in sorted(set(codes or list(LCSC.values()) + [v[1] for v in ORDER.values()])):
        fn = os.path.join(LIBDIR, f"{c}.json")
        if not os.path.exists(fn):
            req = urllib.request.Request(f"https://easyeda.com/api/products/{c}/components?version=6.4.19.5", headers={"User-Agent": "footprint-check/1.0"})
            open(fn, "wb").write(urllib.request.urlopen(req, timeout=30).read())


def _load_fp(code):
    """Pads of a library footprint in mm, part seen from above, +y up, origin at the library origin."""
    U = 0.254
    d = json.load(open(os.path.join(LIBDIR, f"{code}.json")))["result"]
    ds = d["packageDetail"]["dataStr"]
    hx, hy = float(ds["head"]["x"]), float(ds["head"]["y"])
    pads, holes = [], []
    for s in ds["shape"]:
        f = s.split("~")
        if f[0] == "PAD":
            pts = np.array([float(v) for v in f[10].split()]).reshape(-1, 2) if f[10].strip() else np.zeros((0, 2))
            pts = np.stack([(pts[:, 0] - hx) * U, -(pts[:, 1] - hy) * U], 1) if len(pts) else pts
            x, y, w, h = (float(f[2]) - hx) * U, -(float(f[3]) - hy) * U, float(f[4]) * U, float(f[5]) * U
            if f[1] == "POLYGON":
                x, y = pts.mean(0); w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            elif len(pts) == 4:                        # rectangle given by its corners: take size from them (covers rotated pads)
                w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            elif f[1] in ("OVAL", "ELLIPSE") and str(f[11]).strip() in ("90", "270"):
                w, h = h, w
            pads.append(dict(num=int(f[8]), x=float(x), y=float(y), w=float(w), h=float(h), shape={"RECT": "R", "OVAL": "O", "ELLIPSE": "O", "POLYGON": "P"}[f[1]],
                             poly=[[float(a), float(b)] for a, b in pts] if f[1] == "POLYGON" else None))
        elif f[0] == "HOLE":
            holes.append(((float(f[1]) - hx) * U, -(float(f[2]) - hy) * U, float(f[3]) * U))
    pads.sort(key=lambda p: p["num"])
    return dict(pads=pads, holes=holes, name=d.get("title"), package=d["packageDetail"].get("title"))


def _make_footprints():
    out = {}
    for part, code in LCSC.items():
        fp = _load_fp(code)
        (bw, bh, bz), col, pkg = BODY[part]
        xs = [p["x"] - p["w"] / 2 for p in fp["pads"]] + [p["x"] + p["w"] / 2 for p in fp["pads"]] + [-bw / 2, bw / 2]
        ys = [p["y"] - p["h"] / 2 for p in fp["pads"]] + [p["y"] + p["h"] / 2 for p in fp["pads"]] + [-bh / 2, bh / 2]
        m = 0.30 if part in ("C0402", "R0402", "C0603", "R0603", "LED0603", "C0805", "C1206", "SOT23") else 0.12      # small parts keep room for a via beside each pad
        out[part] = dict(pkg=pkg, lcsc=code, lib_name=fp["name"], lib_package=fp["package"], pads=fp["pads"], holes=fp["holes"],
                         box=(min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m), body=(bw, bh, bz), col=col)
    out["TP"] = dict(pkg="test pad 1.0 mm", pads=[dict(num=1, x=0, y=0, w=1.0, h=1.0, shape="C", poly=None)], holes=[], box=(-0.7, 0.7, -0.7, 0.7), body=(1.0, 1.0, 0.03), col="#d4af37", round=True)
    out["FID"] = dict(pkg="fiducial 1.0 mm, 2.0 mm mask opening", pads=[dict(num=1, x=0, y=0, w=1.0, h=1.0, shape="C", poly=None)], holes=[], box=(-1.2, 1.2, -1.2, 1.2),
                      body=(1.0, 1.0, 0.03), col="#d4af37", round=True)
    return out


try:
    FOOTPRINTS = _make_footprints()
except FileNotFoundError:                                   # first run on a fresh checkout
    fetch_lib()
    FOOTPRINTS = _make_footprints()

# ---- stroke font for the silkscreen (4 wide x 6 tall grid) ---------------------
FONT = {
    "0": [[(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)]], "1": [[(1, 5), (2, 6), (2, 0)]],
    "2": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)]],
    "3": [[(0, 6), (4, 6), (2, 3.5), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(4, 6), (0, 6), (0, 3.5), (3, 3.5), (4, 2.5), (4, 1), (3, 0), (0, 0)]],
    "6": [[(4, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.5), (3, 3.5), (0, 3.5)]],
    "7": [[(0, 6), (4, 6), (1.5, 0)]],
    "8": [[(1, 3.5), (0, 4.5), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4.5), (3, 3.5), (1, 3.5), (0, 2.5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.5), (3, 3.5)]],
    "9": [[(0, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 3.5), (1, 2.5), (4, 2.5)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]], "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "D": [[(0, 0), (0, 6), (2.5, 6), (4, 4.5), (4, 1.5), (2.5, 0), (0, 0)]], "J": [[(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]], "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 2.5)], [(1.5, 3.5), (4, 0)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "R": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(2, 3), (4, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]], "V": [[(0, 6), (2, 0), (4, 6)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]], "Q": [[(0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0), (0, 1)], [(2.5, 1.5), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]], "P": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]],
    "-": [[(0.5, 3), (3.5, 3)]], " ": [],
}


class Board:
    G = 0.05                                               # placement raster [mm]
    CLR = 0.20                                             # courtyard to courtyard
    CLR_TH = 0.45                                          # courtyard to a through-hole pad that is hand-soldered later
    CLR_EDGE = 0.30

    def __init__(self, n=5, pitch=10.5, side=58.0, chamfer=1.0, v_drive=18.0):
        self.n, self.pitch, self.side, self.chamfer, self.v_drive = n, pitch, side, chamfer, v_drive
        self.items = []
        self.nets = {}
        self.unplaced = []
        self.N = int(round(side / self.G)) + 1
        self._occ = np.zeros((self.N, self.N), bool)       # back side: where no further courtyard may go
        self._place()
        self._netlist()

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def xf(it, px, py):
        """Part frame (seen from the part's own side) -> board coordinates (seen from the front)."""
        r = it["rot"] % 360
        rx, ry = {0: (px, py), 90: (-py, px), 180: (-px, -py), 270: (py, -px)}[r]
        if it["side"] == "back":
            rx = -rx
        return it["x"] + rx, it["y"] + ry

    def pads_of(self, it):
        """Copper pads of a placed part: dicts with num, x, y, w, h (board axes), shape, poly."""
        fp = FOOTPRINTS[it["part"]]
        out = []
        for p in fp["pads"]:
            x, y = self.xf(it, p["x"], p["y"])
            w, h = (p["h"], p["w"]) if it["rot"] in (90, 270) else (p["w"], p["h"])
            poly = [list(self.xf(it, a, b)) for a, b in p["poly"]] if p["poly"] else None
            out.append(dict(num=p["num"], x=x, y=y, w=w, h=h, shape=p["shape"], poly=poly, ref=it["ref"]))
        return out

    def pad(self, ref, num):
        it = self.item(ref)
        return next(p for p in self.pads_of(it) if p["num"] == num)

    def item(self, ref):
        return next(i for i in self.items if i["ref"] == ref)

    def _box(self, it):
        x0, x1, y0, y1 = FOOTPRINTS[it["part"]]["box"]
        c = [self.xf(it, a, b) for a in (x0, x1) for b in (y0, y1)]
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        return min(xs), max(xs), min(ys), max(ys)

    def _ij(self, v):
        return int(round((v + self.side / 2) / self.G))

    def _mark_rect(self, x0, x1, y0, y1, grow):
        a, b = max(self._ij(x0 - grow), 0), min(self._ij(x1 + grow) + 1, self.N)
        c, d = max(self._ij(y0 - grow), 0), min(self._ij(y1 + grow) + 1, self.N)
        self._occ[c:d, a:b] = True

    def _mark_disc(self, x, y, r):
        a, b = max(self._ij(x - r), 0), min(self._ij(x + r) + 1, self.N)
        c, d = max(self._ij(y - r), 0), min(self._ij(y + r) + 1, self.N)
        yy, xx = np.mgrid[c:d, a:b]
        self._occ[c:d, a:b] |= ((xx * self.G - self.side / 2 - x) ** 2 + (yy * self.G - self.side / 2 - y) ** 2) <= r * r

    def _free(self, x0, x1, y0, y1):
        a, b, c, d = self._ij(x0), self._ij(x1) + 1, self._ij(y0), self._ij(y1) + 1
        if a < 0 or c < 0 or b > self.N or d > self.N:
            return False
        return not self._occ[c:d, a:b].any()

    def _add(self, ref, part, side, x, y, rot=0, d=None, check=True, **kw):
        it = dict(ref=ref, part=part, side=side, x=float(x), y=float(y), rot=int(rot))
        it.update(kw)
        if d is not None:
            it["d"] = float(d)
        else:
            x0, x1, y0, y1 = self._box(it)
            it["cx"], it["cy"], it["w"], it["h"] = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            if check and not self._free(x0, x1, y0, y1):
                self.unplaced.append(ref + " (fixed position is not free)")
            self._mark_rect(x0, x1, y0, y1, self.CLR)
        self.items.append(it)
        return it

    # ------------------------------------------------------------------ placement
    def _place(self):
        n, p, T, h = self.n, self.pitch, TRANSDUCER, self.side / 2
        c = (np.arange(n) - (n - 1) / 2) * p
        self.cols = c
        g = np.arange(self.N) * self.G - h
        X, Y = np.meshgrid(g, g)
        self._occ |= (np.abs(X) > h - self.CLR_EDGE) | (np.abs(Y) > h - self.CLR_EDGE) | ((np.abs(X) + np.abs(Y)) > 2 * h - self.chamfer - self.CLR_EDGE * np.sqrt(2))
        k = 0
        for y in c:
            for x in c:
                k += 1
                self._add(f"T{k}", "TRANSDUCER", "front", x, y, d=T["can_d"])
                for j, sx in ((1, -1), (2, 1)):
                    self._add(f"T{k}.{j}", "PAD", "both", x + sx * T["lead_pitch"] / 2, y, d=T["pad_d"], drill=T["drill"])
                    self._mark_disc(x + sx * T["lead_pitch"] / 2, y, T["pad_d"] / 2 + self.CLR_TH)
        self.n_ch = k
        hm = h - 2.4
        for q, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            self._add(f"H{q+1}", "MOUNT", "both", sx * hm, sy * hm, d=4.4, drill=2.2)
            self._mark_disc(sx * hm, sy * hm, 2.2 + 0.25)
        y0, y1, ys = 5.25, -5.25, 15.75
        # RP2040 turned so that its flash / supply row faces the flash and its crystal row faces the crystal; the two GPIO-only
        # rows face the transducer rows and escape through the gaps between the pads of T13 and T18, which are kept free of parts
        self._add("U1", "RP2040", "back", 0.0, y0, rot=90, check=False)
        for yy in (-3.2, 9.3):
            self._mark_rect(-2.0, 2.0, yy, yy + 4.4, 0.0)
        self._add("U2", "FLASH", "back", 9.4, y0, rot=270)               # this way round the six QSPI lines reach it without crossing
        self._add("U3", "RS485", "back", 17.2, y0)
        self._add("U4", "LDO", "back", -17.6, y0)
        self._add("Y1", "XTAL", "back", -6.4, y0)
        # microphone: its sound hole must sit in the gap between four cans
        hx, hy, hr = FOOTPRINTS["MIC"]["holes"][0]
        gap = np.array([(c[0] + c[1]) / 2, y1])
        mk = dict(ref="MK1", part="MIC", side="back", x=0.0, y=0.0, rot=0)
        ox, oy = self.xf(mk, hx, hy)
        mic = self._add("MK1", "MIC", "back", gap[0] - ox, gap[1] - oy, hole=2 * hr, hole_xy=[float(gap[0]), float(gap[1])])
        # the library draws the ground ring round the sound hole as four separate arcs (for the solder paste); join them with a
        # copper ring under the solder mask so that the ring is one pad electrically
        rr = [np.hypot(a - gap[0], c - gap[1]) for q in self.pads_of(mic) if q["poly"] for a, c in q["poly"]]
        mic["ring"] = dict(r=float((min(rr) + max(rr)) / 2), w=float(0.6 * (max(rr) - min(rr))), pad=3)
        # drivers, each next to the two transducers it drives
        self.drv_of = {}
        q = 0
        for i, x in enumerate(c):                                      # lower strip: rows 1 and 2
            q += 1; self._add(f"D{q}", "DRV", "back", x, -ys, rot=180); self.drv_of[q] = [1 + i, 6 + i]
        for i, x in enumerate(c):                                      # upper strip: rows 4 and 5
            q += 1; self._add(f"D{q}", "DRV", "back", x - (4.8 if i == 2 else 0.0), ys, rot=0); self.drv_of[q] = [16 + i, 21 + i]
        for x, ts in ((c[1], [11, 12]), (c[2] + 4.8, [13]), (c[3], [14, 15])):    # middle row; the centre one is moved off the escape channel
            q += 1; self._add(f"D{q}", "DRV", "back", x, y1, rot=0); self.drv_of[q] = ts
        self.n_drv = q
        self._add("J1", "J_BUS", "back", -6.0, 25.6, rot=180)
        self._add("J2", "J_BUS", "back", 6.0, 25.6, rot=180)
        self._add("J3", "J_PWR", "back", -23.2, y1 - 0.65)                    # top-entry power connectors at the two ends of the lower middle strip
        self._add("J4", "J_PWR", "back", 20.2, y1 - 0.65)
        self._add("J5", "J_SWD", "back", -25.5, y0, rot=90)            # programming connector on the side of the SWD pins, opening towards the board edge
        # passives: (ref, footprint, value key, anchor (ref, pad), which of its own pads should be nearest the anchor)
        want = [("CB1", "C1206", "10uF50", ("J3", 1), 1), ("CB2", "C1206", "10uF50", ("J4", 1), 1)]
        want += [(f"CD{q}", "C0603", "1uF50", (f"D{q}", 6), 1) for q in range(1, self.n_drv + 1)]
        want += [("C10", "C0402", "1uF", ("U1", 45), 1), ("C9", "C0402", "1uF", ("U1", 44), 1), ("C18", "C0402", "100nF", ("U1", 23), 1), ("C19", "C0402", "100nF", ("U1", 50), 1),
                 ("C11", "C0402", "15pF", ("Y1", 1), 1), ("R1", "R0402", "1k", ("Y1", 3), 2), ("C12", "C0402", "15pF", ("Y1", 3), 1)]
        want += [(f"C{i+1}", "C0402", "100nF", ("U1", pin), 1) for i, pin in enumerate((1, 10, 22, 33, 42, 49, 43, 48))]
        want += [("C13", "C0402", "100nF", ("U2", 8), 1), ("R2", "R0402", "10k", ("U2", 1), 1),
                 ("C14", "C0805", "10uF", ("U4", 3), 1), ("C15", "C0805", "10uF", ("U4", 2), 1), ("C20", "C0805", "10uF", ("U4", 4), 1),
                 ("C16", "C0402", "100nF", ("U3", 8), 1), ("Q1", "SOT23", "SOT23", ("U3", 3), 3), ("R8", "R0402", "10k", ("U3", 3), 2), ("R7", "R0402", "10k", ("Q1", 1), 2),
                 ("R3", "R0603", "120R", ("U3", 6), 1), ("C17", "C0402", "100nF", ("MK1", 5), 1),
                 ("R4", "R0402", "100R", ("J1", 5), 1), ("R5", "R0402", "10k", ("R4", 2), 1), ("LD1", "LED0603", "LED", ("U4", 4), 1), ("R6", "R0402", "1k", ("LD1", 1), 2),
                 ("TP4", "TP", "RUN test pad", ("U1", 26), 1)]
        for ref, part, val, anchor, near in want:
            a = self.pad(*anchor)
            if not self._autoplace(ref, part, a["x"], a["y"], near, value=val):
                self.unplaced.append(ref)
        for q, (fx, fy) in enumerate(((-24.5, 26.3), (24.5, -26.0), (24.5, 26.0))):
            if not self._autoplace(f"FID{q+1}", "FID", fx, fy, 1, value="fiducial", r_max=7.0):
                self.unplaced.append(f"FID{q+1}")

    def _autoplace(self, ref, part, ax, ay, near_pad, value="", r_max=12.0, step=0.25):
        """Put a small part where its pad `near_pad` is as close as possible to (ax, ay)."""
        best = None
        g = np.arange(-r_max, r_max + 1e-9, step)
        GX, GY = np.meshgrid(g, g)
        sc = np.hypot(GX, GY)
        order = np.argsort(sc, axis=None)
        for rot in (0, 90, 180, 270):
            it = dict(ref=ref, part=part, side="back", x=0.0, y=0.0, rot=rot)
            p = next(q for q in self.pads_of(it) if q["num"] == near_pad)
            x0, x1, y0, y1 = self._box(it)
            ox, oy = ax + GX - p["x"], ay + GY - p["y"]             # part origin such that the near pad sits at anchor + (dx, dy)
            for k in order:
                if best is not None and sc.flat[k] >= best[0]:
                    break
                px, py = ox.flat[k], oy.flat[k]
                if self._free(px + x0, px + x1, py + y0, py + y1):
                    best = (sc.flat[k], rot, px, py)
                    break
        if best is None:
            return False
        self._add(ref, part, "back", best[2], best[3], rot=best[1], value=value, check=False)
        return True

    # ------------------------------------------------------------------ netlist
    def _netlist(self):
        """self.padnet[(ref, num)] = net name (None = not connected); self.nets[name] = [(ref, num), ...]."""
        pn = {}
        two = {"CB": ("+VDRIVE", "GND"), "CD": ("+VDRIVE", "GND"), "C9": ("+3V3", "GND"), "C10": ("+1V1", "GND"), "C18": ("+1V1", "GND"), "C19": ("+1V1", "GND"),
               "C11": ("XIN", "GND"), "C12": ("XTAL2", "GND"), "R1": ("XTAL2", "XOUT"), "C13": ("+3V3", "GND"), "R2": ("QSPI_SS", "+3V3"),
               "C14": ("+5V", "GND"), "C15": ("+3V3", "GND"), "C20": ("+3V3", "GND"), "C16": ("+3V3", "GND"), "R3": ("RS485_A", "RS485_B"), "R7": ("QB", "UART_TX"),
               "R8": ("+3V3", "DE"), "C17": ("+3V3", "GND"), "R4": ("SYNC_BUS", "SYNC"), "R5": ("SYNC", "GND"), "LD1": ("LED_A", "GND"), "R6": ("+3V3", "LED_A")}
        # which driver output feeds which transducer: the nearer output pad
        ch = {}                                             # transducer k -> (driver ref, "A" | "B")
        for q, ts in self.drv_of.items():
            pa, pb = self.pad(f"D{q}", 7), self.pad(f"D{q}", 5)
            if len(ts) == 1:
                ch[ts[0]] = (f"D{q}", "A")
            else:
                t0, t1 = [self.item(f"T{t}.1") for t in ts]
                da = np.hypot(pa["x"] - t0["x"], pa["y"] - t0["y"]) + np.hypot(pb["x"] - t1["x"], pb["y"] - t1["y"])
                db = np.hypot(pa["x"] - t1["x"], pa["y"] - t1["y"]) + np.hypot(pb["x"] - t0["x"], pb["y"] - t0["y"])
                a_t, b_t = (ts[0], ts[1]) if da <= db else (ts[1], ts[0])
                ch[a_t] = (f"D{q}", "A"); ch[b_t] = (f"D{q}", "B")
        # which GPIO feeds which driver input: the assignment that keeps the fan-out untangled
        from scipy.optimize import linear_sum_assignment
        u1 = {p["num"]: p for p in self.pads_of(self.item("U1"))}
        gp = [(name, u1[num]) for num, name in PINS["RP2040"].items() if name in DRIVE_GPIO]
        tk = sorted(ch)
        cost = np.zeros((len(tk), len(gp)))
        cu = np.array([self.item("U1")["x"], self.item("U1")["y"]])
        for a, t in enumerate(tk):
            d, o = ch[t]
            ip = self.pad(d, 2 if o == "A" else 4)
            v = np.array([ip["x"], ip["y"]]) - cu
            for b, (name, p) in enumerate(gp):
                w = np.array([p["x"], p["y"]]) - cu
                ang = np.arccos(np.clip(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)), -1, 1))
                cost[a, b] = ang * 6.0 + 0.05 * np.linalg.norm(v - w)
        ra, rb = linear_sum_assignment(cost)
        self.channel = {}
        gpio_net = {}
        for a, b in zip(ra, rb):
            t = tk[a]; d, o = ch[t]
            self.channel[t] = dict(gpio=int(gp[b][0][4:]), driver=d, output=o)
            gpio_net[gp[b][0]] = f"PH{t}"
        for it in self.items:
            part, ref = it["part"], it["ref"]
            if part in ("TRANSDUCER", "MOUNT"):
                continue
            if part == "PAD":
                t, j = ref[1:].split(".")
                pn[(ref, 1)] = f"OUT{t}" if j == "1" else "GND"
            elif part == "RP2040":
                for num, name in PINS[part].items():
                    pn[(ref, num)] = gpio_net.get(name, RP2040_NET.get(name, name))
            elif part == "DRV":
                ts = {v["output"]: t for t, v in self.channel.items() if v["driver"] == ref}
                pn.update({(ref, 1): None, (ref, 8): None, (ref, 3): "GND", (ref, 6): "+VDRIVE",
                           (ref, 2): f"PH{ts['A']}" if "A" in ts else "GND", (ref, 7): f"OUT{ts['A']}" if "A" in ts else None,
                           (ref, 4): f"PH{ts['B']}" if "B" in ts else "GND", (ref, 5): f"OUT{ts['B']}" if "B" in ts else None})
            elif part in PINS:
                for num, name in PINS[part].items():
                    pn[(ref, num)] = name
            elif part == "TP":
                pn[(ref, 1)] = "RUN"
            elif part == "FID":
                pn[(ref, 1)] = None
            else:
                key = ref if ref in two else ref.rstrip("0123456789")
                a, b = two[key] if key in two else ("+3V3", "GND")          # C1..C8: RP2040 supply bypass
                pn[(ref, 1)], pn[(ref, 2)] = a, b
        self.padnet = pn
        nets = {}
        for (ref, num), name in pn.items():
            if name:
                nets.setdefault(name, []).append((ref, num))
        self.nets = nets

    def net_summary(self):
        return {k: [f"{r}.{n}" for r, n in v] for k, v in sorted(self.nets.items())}

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _gap(a, b):
        """Clearance between two courtyards (circle or axis-aligned rectangle); negative = overlap."""
        def rect(it):
            return it["cx"] - it["w"] / 2, it["cx"] + it["w"] / 2, it["cy"] - it["h"] / 2, it["cy"] + it["h"] / 2
        ca, cb = "d" in a, "d" in b
        if ca and cb:
            return np.hypot(a["x"] - b["x"], a["y"] - b["y"]) - (a["d"] + b["d"]) / 2
        if ca != cb:
            c, r = (a, b) if ca else (b, a)
            x0, x1, y0, y1 = rect(r)
            dx = max(x0 - c["x"], 0, c["x"] - x1); dy = max(y0 - c["y"], 0, c["y"] - y1)
            if dx == 0 and dy == 0:
                return -min(c["x"] - x0, x1 - c["x"], c["y"] - y0, y1 - c["y"]) - c["d"] / 2
            return np.hypot(dx, dy) - c["d"] / 2
        ax0, ax1, ay0, ay1 = rect(a); bx0, bx1, by0, by1 = rect(b)
        dx = max(bx0 - ax1, ax0 - bx1); dy = max(by0 - ay1, ay0 - by1)
        if dx < 0 and dy < 0:
            return max(dx, dy)
        return float(np.hypot(max(dx, 0), max(dy, 0)))

    def _edge_gap(self, it):
        h = self.side / 2
        if "d" in it:
            ex = h - abs(it["x"]) - it["d"] / 2; ey = h - abs(it["y"]) - it["d"] / 2
            ch = (2 * h - self.chamfer - abs(it["x"]) - abs(it["y"])) / np.sqrt(2) - it["d"] / 2
        else:
            ex = h - abs(it["cx"]) - it["w"] / 2; ey = h - abs(it["cy"]) - it["h"] / 2
            ch = (2 * h - self.chamfer - (abs(it["cx"]) + it["w"] / 2) - (abs(it["cy"]) + it["h"] / 2)) / np.sqrt(2)
        return min(ex, ey, ch)

    def drc(self, clr_part=0.18, clr_pad=0.40, clr_edge=0.25, clr_can=0.30):
        """Placement rules (the copper has its own check in pcbroute).  Returns (violations, minimum clearances).
        front: can-to-can and can-to-mounting-hole >= clr_can; the microphone port >= 1 mm from any can;
        back : every courtyard vs every other, vs every hand-soldered through-hole pad, vs the mounting-hole keep-outs;
        all  : everything >= clr_edge from the board edge, drilled holes >= 0.5 mm."""
        v = []
        cans = [i for i in self.items if i["part"] == "TRANSDUCER"]
        pads = [i for i in self.items if i["part"] == "PAD"]
        holes = [i for i in self.items if i["part"] == "MOUNT"]
        back = [i for i in self.items if i["side"] == "back"]
        mins = dict(can_can=np.inf, can_hole=np.inf, part_part=np.inf, part_pad=np.inf, pad_pad=np.inf, edge=np.inf, hole_edge=np.inf, mic_port_can=np.inf)
        for a in range(len(cans)):
            for b in range(a + 1, len(cans)):
                g = self._gap(cans[a], cans[b]); mins["can_can"] = min(mins["can_can"], g)
                if g < clr_can:
                    v.append(f"cans {cans[a]['ref']}-{cans[b]['ref']} gap {g:.2f} < {clr_can}")
            for hh in holes:
                g = np.hypot(cans[a]["x"] - hh["x"], cans[a]["y"] - hh["y"]) - cans[a]["d"] / 2 - hh["drill"] / 2
                mins["can_hole"] = min(mins["can_hole"], g)
                if g < clr_can:
                    v.append(f"can {cans[a]['ref']} to hole {hh['ref']} gap {g:.2f} < {clr_can}")
            for mk in [i for i in self.items if i.get("hole")]:
                g = np.hypot(cans[a]["x"] - mk["hole_xy"][0], cans[a]["y"] - mk["hole_xy"][1]) - cans[a]["d"] / 2 - mk["hole"] / 2
                mins["mic_port_can"] = min(mins["mic_port_can"], g)
                if g < 1.0:
                    v.append(f"microphone port under/next to can {cans[a]['ref']} (gap {g:.2f} < 1.0)")
        for a in range(len(back)):
            for b in range(a + 1, len(back)):
                g = self._gap(back[a], back[b]); mins["part_part"] = min(mins["part_part"], g)
                if g < clr_part:
                    v.append(f"back parts {back[a]['ref']}-{back[b]['ref']} gap {g:.2f} < {clr_part}")
            for pd in pads + holes:
                g = self._gap(back[a], pd); mins["part_pad"] = min(mins["part_pad"], g)
                if g < ((0.30 if back[a]["ref"] == "U1" else clr_pad) if pd["part"] == "PAD" else 0.0):
                    v.append(f"{back[a]['ref']} too close to {pd['ref']}: gap {g:.2f}")
        for a in range(len(pads)):
            for b in range(a + 1, len(pads)):
                g = self._gap(pads[a], pads[b]); mins["pad_pad"] = min(mins["pad_pad"], g)
                if g < 0.3:
                    v.append(f"pads {pads[a]['ref']}-{pads[b]['ref']} gap {g:.2f} < 0.3")
        for it in self.items:
            g = self._edge_gap(it)
            if it["part"] == "MOUNT":
                ge = self._edge_gap(dict(x=it["x"], y=it["y"], d=it["drill"]))
                mins["hole_edge"] = min(mins["hole_edge"], ge)
                if ge < 0.5:
                    v.append(f"{it['ref']} hole wall to board edge {ge:.2f} < 0.5")
                continue
            mins["edge"] = min(mins["edge"], g)
            if g < clr_edge:
                v.append(f"{it['ref']} to board edge {g:.2f} < {clr_edge}")
        for r in self.unplaced:
            v.append(f"{r} could not be placed")
        return v, {k: round(float(x), 2) for k, x in mins.items()}

    def electrical(self, duty=1.0):
        T, P = TRANSDUCER, PARTS
        f = 40e3
        p_cv2f = T["cap_nF"] * 1e-9 * self.v_drive ** 2 * f
        p_motional = 0.035 * (self.v_drive / 20.0) ** 2
        p_ch = (p_cv2f + p_motional) * duty
        return dict(
            channels=self.n_ch, gpio_needed=self.n_ch + 1 + 2 + 2, gpio_available=P["RP2040"]["gpio"],
            gpio_use=f"{self.n_ch} drive (GPIO4-28) + SYNC (GPIO29) + UART (GPIO0/1) + microphone (GPIO2/3); RS-485 direction comes from TX, the LED is a power LED",
            gpio_ok=self.n_ch + 5 <= P["RP2040"]["gpio"],
            drive_vpp=self.v_drive, transducer_max_vpp=T["v_max_pp"], transducer_ok=self.v_drive <= T["v_max_pp"],
            driver_max_v=P["DRV"]["v_max"], driver_ok=self.v_drive <= 0.9 * P["DRV"]["v_max"],
            driver_headroom_pct=round(100 * (1 - self.v_drive / P["DRV"]["v_max"]), 1),
            driver_peak_current_a=P["DRV"]["i_peak"],
            edge_charge_time_ns=round(T["cap_nF"] * 1e-9 * self.v_drive / P["DRV"]["i_peak"] * 1e9, 1),
            phase_step_ns_at_128=round(1e9 / f / 128, 1),
            power_per_channel_mw=round(p_ch * 1e3, 1), board_power_w=round(p_ch * self.n_ch + 0.15, 2),
            board_current_a=round((p_ch * self.n_ch) / self.v_drive, 3), drivers=self.n_drv,
            driver_dissipation_mw=round(2 * p_cv2f * 1e3 * duty, 1),
            pio_sample_rate_mhz=round(f * 128 / 1e6, 2), dma_words_per_period=128)

    # ------------------------------------------------------------------ outputs
    def model(self):
        """Component bodies for the 3-D model: ref, part, x, y, w, h (board axes), height, colour."""
        out = []
        for it in self.items:
            if it["part"] in ("PAD", "TRANSDUCER", "MOUNT"):
                continue
            fp = FOOTPRINTS[it["part"]]
            bw, bh, bz = fp["body"]
            if it["rot"] in (90, 270):
                bw, bh = bh, bw
            out.append(dict(ref=it["ref"], part=it["part"], x=it["x"], y=it["y"], w=bw, h=bh, z=bz, col=fp["col"], round=bool(fp.get("round"))))
        return out

    def to_json(self, path, extra=None):
        v, mins = self.drc()
        data = dict(name=f"TB-{self.n_ch}", side_mm=self.side, chamfer_mm=self.chamfer, thickness_mm=1.6, layers=4,
                    items=self.items, bodies=self.model(), nets=self.net_summary(), channel_map={str(k): c for k, c in sorted(self.channel.items())},
                    drc=dict(violations=v, min_clearances_mm=mins), electrical=self.electrical(), transducer=TRANSDUCER,
                    footprints={k: dict(pkg=f["pkg"], lcsc=f.get("lcsc"), library_package=f.get("lib_package"), body=f["body"]) for k, f in FOOTPRINTS.items()})
        data.update(extra or {})
        json.dump(data, open(path, "w"), indent=1)
        return data

    def outline(self):
        h, ch = self.side / 2, self.chamfer
        return [(-h + ch, -h), (h - ch, -h), (h, -h + ch), (h, h - ch), (h - ch, h), (-h + ch, h), (-h, h - ch), (-h, -h + ch), (-h + ch, -h)]

    def text(self, s, x, y, size, mirror):
        """Stroke text centred on (x, y); mirrored for the back so it reads from that side."""
        sc = size / 6.0
        adv = 5.2 * sc
        w = adv * len(s) - 1.2 * sc
        out = []
        for i, chh in enumerate(s):
            for poly in FONT.get(chh, []):
                pts = []
                for (gx, gy) in poly:
                    lx = -w / 2 + i * adv + gx * sc
                    pts.append((x - lx if mirror else x + lx, y - size / 2 + gy * sc))
                out.append(pts)
        return out

    def to_svg(self, path, side="back", tracks=None, vias=None):
        s = self.side
        sc = 14.0
        W = int((s + 8) * sc)
        mir = side == "back"
        def X(x): return ((-x if mir else x) + s / 2 + 4) * sc
        def Y(y): return (s / 2 + 4 - y) * sc
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}" font-family="sans-serif">',
             f'<rect width="{W}" height="{W}" fill="#0f172a"/>',
             '<polygon points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in self.outline()) + '" fill="#14532d" stroke="#86efac" stroke-width="1.5"/>']
        for tr in tracks or []:
            if tr["layer"] == "L3" or (tr["layer"] == "L4") == (side == "back"):
                col = "#f59e0b" if tr["layer"] == "L4" else "#38bdf8"
                o.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in tr["pts"]) + f'" fill="none" stroke="{col}" stroke-opacity="{0.95 if tr["layer"] == "L4" else 0.6}" '
                         f'stroke-width="{tr["w"]*sc:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
        for vv in vias or []:
            o.append(f'<circle cx="{X(vv["x"]):.1f}" cy="{Y(vv["y"]):.1f}" r="{0.3*sc:.1f}" fill="#e2e8f0"/><circle cx="{X(vv["x"]):.1f}" cy="{Y(vv["y"]):.1f}" r="{0.15*sc:.1f}" fill="#0f172a"/>')
        for it in self.items:
            if it["part"] == "PAD":
                o.append(f'<circle cx="{X(it["x"]):.1f}" cy="{Y(it["y"]):.1f}" r="{it["d"]/2*sc:.1f}" fill="#eab308"/>'
                         f'<circle cx="{X(it["x"]):.1f}" cy="{Y(it["y"]):.1f}" r="{it["drill"]/2*sc:.1f}" fill="#0f172a"/>')
            elif it["part"] == "MOUNT":
                o.append(f'<circle cx="{X(it["x"]):.1f}" cy="{Y(it["y"]):.1f}" r="{it["d"]/2*sc:.1f}" fill="none" stroke="#fbbf24" stroke-dasharray="4 3"/>'
                         f'<circle cx="{X(it["x"]):.1f}" cy="{Y(it["y"]):.1f}" r="{it["drill"]/2*sc:.1f}" fill="#0f172a" stroke="#e2e8f0"/>')
        if side == "front":
            for it in self.items:
                if it["part"] == "TRANSDUCER":
                    o.append(f'<circle cx="{X(it["x"]):.1f}" cy="{Y(it["y"]):.1f}" r="{it["d"]/2*sc:.1f}" fill="#c9ccd1" fill-opacity="0.85" stroke="#e2e8f0"/>'
                             f'<text x="{X(it["x"]):.1f}" y="{Y(it["y"])+4:.1f}" font-size="11" fill="#0f172a" text-anchor="middle">{it["ref"]}</text>')
        else:
            for b in self.model():
                x, y = X(b["x"]), Y(b["y"])
                if not b["round"]:
                    o.append(f'<rect x="{x-b["w"]/2*sc:.1f}" y="{y-b["h"]/2*sc:.1f}" width="{b["w"]*sc:.1f}" height="{b["h"]*sc:.1f}" rx="1.5" '
                             f'fill="{b["col"]}" fill-opacity="0.55" stroke="#e2e8f0" stroke-width="0.4"/>')
            for it in self.items:
                if it["side"] != "back":
                    continue
                for p in self.pads_of(it):
                    if p["shape"] == "C":
                        o.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="{p["w"]/2*sc:.1f}" fill="#eab308"/>')
                    elif p["poly"]:
                        o.append('<polygon points="' + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in p["poly"]) + '" fill="#eab308"/>')
                    else:
                        o.append(f'<rect x="{X(p["x"])-p["w"]/2*sc:.1f}" y="{Y(p["y"])-p["h"]/2*sc:.1f}" width="{p["w"]*sc:.1f}" height="{p["h"]*sc:.1f}" '
                                 f'rx="{(min(p["w"], p["h"])/2*sc if p["shape"] == "O" else 0):.1f}" fill="#eab308"/>')
            for b in self.model():
                if b["part"] in ("RP2040", "DRV", "RS485", "FLASH", "J_BUS", "J_PWR", "J_SWD", "LDO", "MIC", "XTAL"):
                    o.append(f'<text x="{X(b["x"]):.1f}" y="{Y(b["y"])+3.5:.1f}" font-size="10" fill="#f8fafc" text-anchor="middle">{b["ref"]}</text>')
        o.append(f'<text x="8" y="16" font-size="13" fill="#e2e8f0">TB-{self.n_ch} {side} side ({"as seen from the back" if mir else "as seen from the front"}) '
                 f'- {s:.0f} x {s:.0f} mm' + (" - orange: back copper, blue: inner layer 3" if tracks else "") + '</text></svg>')
        open(path, "w").write("\n".join(o))


if __name__ == "__main__":
    b = Board()
    v, mins = b.drc()
    n_parts = len([i for i in b.items if i["part"] not in ("PAD", "MOUNT")])
    print(f"TB-25: {n_parts} placed parts ({len([i for i in b.items if i['side']=='back'])} on the back), placement violations: {len(v)}")
    for x in v[:30]:
        print("   ", x)
    print("  minimum clearances (mm):", mins)
    print("  nets:", len(b.nets), "| pads with a net:", sum(len(x) for x in b.nets.values()))
