"""Candidate hardware configurations (transducer layouts).

Coordinates: origin at the centre of the build volume, +z up, the pole enters
from above along the z axis.  All dimensions in metres.
"""
import numpy as np
from .acoustics import Panel, Array
from . import constants as C

POLE_PORT_MM = 4.0e-3     # clearance radius in the top panel for the pole
JET_PORT_MM = 2.5e-3      # clearance radius in the bottom panel for the droplet jet
JET_XY = (0.0315, 0.0)    # jet axis position (centre of a 2x2 transducer gap on even grids)


def _panel_extent(n, pitch=C.PITCH):
    return n * pitch


def opposed(n=16, sep=0.20, vpp=20.0):
    """Two facing flat boards (classic 2x 16x16 levitator)."""
    panels = [
        Panel("bottom", [0, 0, -sep / 2], [0, 0, 1], [1, 0, 0], n, n,
              holes=[(JET_XY[0], JET_XY[1], JET_PORT_MM)]),
        Panel("top", [0, 0, sep / 2], [0, 0, -1], [1, 0, 0], n, n,
              holes=[(0, 0, POLE_PORT_MM)]),
    ]
    return Array(panels, vpp=vpp, name=f"opposed_{n}x{n}_sep{int(sep*1e3)}")


def bottom_only(n=16, dist=0.10, vpp=20.0):
    panels = [Panel("bottom", [0, 0, -dist], [0, 0, 1], [1, 0, 0], n, n,
                    holes=[(JET_XY[0], JET_XY[1], JET_PORT_MM)])]
    return Array(panels, vpp=vpp, name=f"bottom_{n}x{n}")


def box(n=12, half=None, top=True, bottom=True, walls=4, vpp=20.0, nz=None):
    """Rectangular enclosure of flat boards: bottom + `walls` side boards (+ top).

    half: distance from the centre to each board face (default: boards just touch).
    nz:   number of transducer rows on the walls in the vertical direction.
    """
    nz = nz or n
    if half is None:
        half = _panel_extent(n) / 2 + 2e-3
    panels = []
    if bottom:
        panels.append(Panel("bottom", [0, 0, -half], [0, 0, 1], [1, 0, 0], n, n,
                            holes=[(JET_XY[0], JET_XY[1], JET_PORT_MM)]))
    wall_defs = [("wall+x", [half, 0, 0], [-1, 0, 0], [0, 1, 0]),
                 ("wall-x", [-half, 0, 0], [1, 0, 0], [0, 1, 0]),
                 ("wall+y", [0, half, 0], [0, -1, 0], [1, 0, 0]),
                 ("wall-y", [0, -half, 0], [0, 1, 0], [1, 0, 0])]
    for nm, c, nr, u in wall_defs[:walls]:
        panels.append(Panel(nm, c, nr, u, n, nz))
    if top:
        panels.append(Panel("top", [0, 0, half], [0, 0, -1], [1, 0, 0], n, n,
                            holes=[(0, 0, POLE_PORT_MM)]))
    nm = f"box{len(panels)}_{n}x{nz}_half{int(round(half*1e3))}"
    return Array(panels, vpp=vpp, name=nm)


def prism(sides=6, n=8, nz=12, top=True, bottom=True, nb=12, vpp=20.0, gap=2e-3):
    """Regular prism: `sides` vertical boards (n wide x nz tall) around the axis,
    closed by square bottom/top boards of nb x nb."""
    wwid = _panel_extent(n) + gap
    apothem = wwid / (2 * np.tan(np.pi / sides))
    hz = _panel_extent(nz) / 2 + gap
    panels = []
    if bottom:
        panels.append(Panel("bottom", [0, 0, -hz], [0, 0, 1], [1, 0, 0], nb, nb,
                            holes=[(JET_XY[0], JET_XY[1], JET_PORT_MM)]))
    for i in range(sides):
        ang = 2 * np.pi * (i + 0.5) / sides
        nr = -np.array([np.cos(ang), np.sin(ang), 0])
        c = -nr * apothem
        u = np.array([-np.sin(ang), np.cos(ang), 0])
        panels.append(Panel(f"wall{i}", c, nr, u, n, nz))
    if top:
        panels.append(Panel("top", [0, 0, hz], [0, 0, -1], [1, 0, 0], nb, nb,
                            holes=[(0, 0, POLE_PORT_MM)]))
    return Array(panels, vpp=vpp, name=f"prism{sides}_{n}x{nz}_ap{int(round(apothem*1e3))}")


# rotation taking the cube body diagonal (1,1,1) to +z, so that a triangular face of the
# polyhedron is on top (pole port) and one is underneath (jet + drip port)
_E1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
_E3 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
_E2 = np.cross(_E3, _E1)
ROT3 = np.stack([_E1, _E2, _E3])


def rco(n=5, edge=None, vpp=18.0, vertical=True):
    """Rhombicuboctahedron: 18 identical square boards (n x n transducers) all facing
    the centre; the 8 triangular faces are left open as ports (pole, jet, cameras, UV).

    edge: polyhedron edge length (default: board = transducer field + 5.5 mm).
    Distance centre -> board face = edge (1 + sqrt 2) / 2.
    vertical=True puts a triangular port straight up and straight down.
    """
    if edge is None:
        edge = n * C.PITCH + 5.5e-3
    r_s = edge * (1 + np.sqrt(2)) / 2
    faces = []
    ax = np.eye(3)
    for k in range(3):
        for sgn in (1, -1):
            faces.append((sgn * ax[k], ax[(k + 1) % 3]))
    for k in range(3):                       # squares between axis faces: normal in the plane perpendicular to axis k
        a, b = ax[(k + 1) % 3], ax[(k + 2) % 3]
        for sa in (1, -1):
            for sb in (1, -1):
                faces.append(((sa * a + sb * b) / np.sqrt(2), ax[k]))
    Rm = ROT3 if vertical else np.eye(3)
    panels = []
    for i, (nrm, u) in enumerate(faces):
        nrm = Rm @ nrm; u = Rm @ u
        panels.append(Panel(f"sq{i:02d}", r_s * nrm, -nrm, u, n, n))
    arr = Array(panels, vpp=vpp, name=f"rco18_{n}x{n}_r{int(round(r_s*1e3))}")
    arr.edge = edge
    arr.r_face = r_s
    arr.r_port = edge * (3 + np.sqrt(2)) / (2 * np.sqrt(3))     # centre -> triangular port plane
    return arr


def sphere(N=450, R=0.065, vpp=18.0, cap_deg=22.0):
    """Idealised spherical shell: N transducers on a Fibonacci lattice, each aimed at
    the centre, with polar caps left open for the pole and the jet.  (Hand-wired dome;
    included as the upper bound that flat boards approximate.)"""
    panels = []
    i = np.arange(N * 2)
    z = 1 - 2 * (i + 0.5) / (N * 2)
    keep = np.abs(z) < np.cos(np.radians(cap_deg))
    idx = i[keep]
    idx = idx[np.round(np.linspace(0, len(idx) - 1, N)).astype(int)]
    ga = np.pi * (3 - np.sqrt(5))
    for j in idx:
        zz = 1 - 2 * (j + 0.5) / (N * 2)
        rr = np.sqrt(1 - zz * zz)
        d = np.array([rr * np.cos(ga * j), rr * np.sin(ga * j), zz])
        u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u)
        panels.append(Panel("s", R * d, -d, u, 1, 1))
    return Array(panels, vpp=vpp, name=f"sphere_{N}_r{int(R*1e3)}")


REGISTRY = {
    "opposed16": lambda vpp=20.0: opposed(16, 0.20, vpp),
    "bottom16": lambda vpp=20.0: bottom_only(16, 0.10, vpp),
    "box5_12": lambda vpp=20.0: box(12, top=False, vpp=vpp),
    "box6_12": lambda vpp=20.0: box(12, top=True, vpp=vpp),
    "box6_16": lambda vpp=20.0: box(16, top=True, vpp=vpp),
    "box6_8": lambda vpp=20.0: box(8, top=True, vpp=vpp),
    "prism6": lambda vpp=20.0: prism(6, 8, 12, vpp=vpp),
    "box6_10": lambda vpp=20.0: box(10, top=True, vpp=vpp),
    "rco4": lambda vpp=18.0: rco(4, vpp=vpp),
    "rco5": lambda vpp=18.0: rco(5, vpp=vpp),
    "rco6": lambda vpp=18.0: rco(6, vpp=vpp),
    "sphere450": lambda vpp=18.0: sphere(450, 0.065, vpp=vpp),
    "sphere300": lambda vpp=18.0: sphere(300, 0.060, vpp=vpp),
}
