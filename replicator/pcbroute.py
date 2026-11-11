"""Copper routing for the TB-25 board, its fabrication / assembly files, and their verification.

Layer stack (JLCPCB 4-layer, 1.6 mm):
  L1 front  ground pour over the whole board (the transducer cans sit on it; no signal runs under a can rim)
  L2        +V_DRIVE plane with a +3V3 island under the microcontroller strip
  L3        signals
  L4 back   all surface-mount parts, short signal traces

A grid router (A* on a 0.05 mm grid, two signal layers, through vias) connects every pad
of the netlist in `pcb.Board`; pads of the three plane nets get a short stub and a via.
Nothing here is trusted on its own word: `verify()` re-reads the Gerber and drill files
that were written, rasterises the copper at 0.02 mm, rebuilds the connectivity from the
artwork alone and compares it with the netlist (opens, shorts), and measures clearances.

Units: millimetres; board origin at the centre, seen from the front.
"""
import json
import os
import numpy as np
from numba import njit
from scipy import ndimage as ndi

from .pcb import Board, FOOTPRINTS, ORDER, TRANSDUCER

G = 0.05                      # routing grid
CLR = 0.15                    # copper to copper, design target
MARGIN = 0.02                 # extra the router keeps on top of CLR
W_SIG, W_PWR, W_OUT = 0.15, 0.20, 0.25
VIA_D, VIA_PAD = 0.30, 0.60
VIA_PITCH = 0.85              # centre to centre, any two vias
EDGE = 0.40                   # copper to board edge
TH_INNER = 0.80               # radius of the inner-layer pad of a transducer hole
ISLAND = (-22.6, 21.3, 1.25, 9.25)        # +3V3 island on L2: x0, x1, y0, y1
PLANE_NETS = ("GND", "+VDRIVE", "+3V3")
L4, L3 = 0, 1


# =============================================================================================
# A* on (layer, y, x)
# =============================================================================================
@njit(cache=True)
def _astar(block, viaok, src, tgt, n, lcost, via_cost, x0, x1, y0, y1, extra, vextra):
    """block[2,n,n] (1 = not allowed), viaok[n,n], src = start node ids, tgt[2,n,n] = goal cells.
    Returns the path as node ids from goal back to a start cell (empty if none)."""
    nn = n * n
    INF = 1e30
    g = np.full(2 * nn, INF, np.float32)
    par = np.full(2 * nn, -1, np.int32)
    closed = np.zeros(2 * nn, np.uint8)
    cap = 6 * nn
    hc = np.empty(cap, np.float32)
    hn = np.empty(cap, np.int32)
    hs = 0
    lmin = min(lcost[0], lcost[1])

    def heur(node):
        r = node % nn
        iy = r // n
        ix = r % n
        dx = 0
        if ix < x0:
            dx = x0 - ix
        elif ix > x1:
            dx = ix - x1
        dy = 0
        if iy < y0:
            dy = y0 - iy
        elif iy > y1:
            dy = iy - y1
        a = dx if dx > dy else dy
        b = dy if dx > dy else dx
        return lmin * (a + 0.41421356 * b)

    for s in src:
        g[s] = 0.0
        hc[hs] = heur(s); hn[hs] = s
        i = hs; hs += 1
        while i > 0:
            p = (i - 1) // 2
            if hc[p] <= hc[i]:
                break
            tc = hc[p]; hc[p] = hc[i]; hc[i] = tc
            tn = hn[p]; hn[p] = hn[i]; hn[i] = tn
            i = p
    dxs = np.array([1, -1, 0, 0, 1, 1, -1, -1])
    dys = np.array([0, 0, 1, -1, 1, -1, 1, -1])
    goal = -1
    while hs > 0:
        node = hn[0]
        hs -= 1
        hc[0] = hc[hs]; hn[0] = hn[hs]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= hs:
                break
            r = l + 1
            m = l
            if r < hs and hc[r] < hc[l]:
                m = r
            if hc[i] <= hc[m]:
                break
            tc = hc[m]; hc[m] = hc[i]; hc[i] = tc
            tn = hn[m]; hn[m] = hn[i]; hn[i] = tn
            i = m
        if closed[node]:
            continue
        closed[node] = 1
        lay = node // nn
        rr = node % nn
        iy = rr // n
        ix = rr % n
        if tgt[lay, iy, ix]:
            goal = node
            break
        gc = g[node]
        for k in range(8):
            jx = ix + dxs[k]
            jy = iy + dys[k]
            if jx < 0 or jy < 0 or jx >= n or jy >= n:
                continue
            if block[lay, jy, jx]:
                continue
            if k >= 4 and (block[lay, iy, jx] or block[lay, jy, ix]):
                continue                                   # no cutting across a blocked corner
            nb = lay * nn + jy * n + jx
            if closed[nb]:
                continue
            c = gc + lcost[lay] * (1.0 if k < 4 else 1.41421356) + extra[lay, jy, jx]
            if c < g[nb]:
                g[nb] = c
                par[nb] = node
                if hs < cap:
                    hc[hs] = c + heur(nb); hn[hs] = nb
                    i = hs; hs += 1
                    while i > 0:
                        p = (i - 1) // 2
                        if hc[p] <= hc[i]:
                            break
                        tc = hc[p]; hc[p] = hc[i]; hc[i] = tc
                        tn = hn[p]; hn[p] = hn[i]; hn[i] = tn
                        i = p
        if viaok[iy, ix]:
            ol = 1 - lay
            if not block[ol, iy, ix]:
                nb = ol * nn + iy * n + ix
                if not closed[nb]:
                    c = gc + via_cost + vextra[iy, ix]
                    if c < g[nb]:
                        g[nb] = c
                        par[nb] = node
                        if hs < cap:
                            hc[hs] = c + heur(nb); hn[hs] = nb
                            i = hs; hs += 1
                            while i > 0:
                                p = (i - 1) // 2
                                if hc[p] <= hc[i]:
                                    break
                                tc = hc[p]; hc[p] = hc[i]; hc[i] = tc
                                tn = hn[p]; hn[p] = hn[i]; hn[i] = tn
                                i = p
    if goal < 0:
        return np.empty(0, np.int32)
    out = []
    node = goal
    while node >= 0:
        out.append(node)
        if g[node] == 0.0:
            break
        node = par[node]
    return np.array(out, np.int32)


@njit(cache=True)
def _stamp(arr, iy, ix, r_cells, val):
    n = arr.shape[0]
    R = int(np.ceil(r_cells))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy <= r_cells * r_cells:
                y = iy + dy; x = ix + dx
                if 0 <= y < n and 0 <= x < n:
                    arr[y, x] = val


class Router:
    """Sequential grid router with rip-up: a net that cannot get through is routed across the nets in its
    way at a penalty, those nets are removed and routed again."""
    PEN = 60.0                                               # cost of a cell that conflicts with another routed net
    MAX_RIPS = 60                                            # a net ripped up this often is left alone afterwards

    def __init__(self, board: Board, verbose=True):
        self.b = board
        self.h = board.side / 2
        self.n = int(round(board.side / G)) + 1
        self.verbose = verbose
        self.netid = {name: i + 1 for i, name in enumerate(sorted(board.nets))}
        self.netname = {v: k for k, v in self.netid.items()}
        n = self.n
        self.static = np.zeros((2, n, n), np.int16)          # pads, holes, keep-outs: net id per node of L4 / L3 (-1 = keep-out)
        self.smd = np.zeros((n, n), bool)
        self.drill_static = np.zeros((n, n), bool)
        self.padcells = {}
        self.holes_static = []
        g = np.arange(n) * G - self.h
        self.X, self.Y = np.meshgrid(g, g)
        self._copper()
        self.routes = {}                                     # job -> dict(net, segs=[(layer, iy[], ix[], w)], vias=[(iy, ix, plane)])
        self.jobid = {}
        self.rips = {}
        self.locked = set()
        self.failed = []
        self.n_ripups = 0
        self.hist = np.zeros((2, n, n), np.float32)          # contested cells get dearer every time a net is ripped up there
        self._rebuild()

    # ----------------------------------------------------------------------------------- set-up
    def ij(self, x, y):
        return int(round((y + self.h) / G)), int(round((x + self.h) / G))

    def _raster_shape(self, p):
        """Nodes covered by a pad (grown by just under half a cell): boolean mask in a window, and the window origin."""
        e = G / 2 - 1e-6
        x0, x1, y0, y1 = p["x"] - p["w"] / 2 - e, p["x"] + p["w"] / 2 + e, p["y"] - p["h"] / 2 - e, p["y"] + p["h"] / 2 + e
        if p.get("poly"):
            xs, ys = [q[0] for q in p["poly"]], [q[1] for q in p["poly"]]
            x0, x1, y0, y1 = min(xs) - e, max(xs) + e, min(ys) - e, max(ys) + e
        a, b = max(int(np.ceil((x0 + self.h) / G)), 0), min(int(np.floor((x1 + self.h) / G)), self.n - 1)
        c, d = max(int(np.ceil((y0 + self.h) / G)), 0), min(int(np.floor((y1 + self.h) / G)), self.n - 1)
        X, Y = self.X[c:d + 1, a:b + 1], self.Y[c:d + 1, a:b + 1]
        if p.get("poly"):
            from matplotlib.path import Path
            pth = Path(np.array(p["poly"]))
            pts = np.stack([X.ravel(), Y.ravel()], 1)
            m = (pth.contains_points(pts, radius=e) | pth.contains_points(pts, radius=-e)).reshape(X.shape)
        elif p["shape"] == "C":
            m = (X - p["x"]) ** 2 + (Y - p["y"]) ** 2 <= (p["w"] / 2 + e) ** 2
        elif p["shape"] == "O":
            r = min(p["w"], p["h"]) / 2
            hx, hy = max(p["w"] / 2 - r, 0), max(p["h"] / 2 - r, 0)
            dx, dy = np.maximum(np.abs(X - p["x"]) - hx, 0), np.maximum(np.abs(Y - p["y"]) - hy, 0)
            m = dx ** 2 + dy ** 2 <= (r + e) ** 2
        else:
            m = np.ones(X.shape, bool)
        return m, c, a

    def _copper(self):
        b = self.b
        edge = (np.abs(self.X) > self.h - EDGE) | (np.abs(self.Y) > self.h - EDGE) | (np.abs(self.X) + np.abs(self.Y) > 2 * self.h - b.chamfer - EDGE * np.sqrt(2))
        self.static[:, edge] = -1
        R2 = lambda x, y: (self.X - x) ** 2 + (self.Y - y) ** 2
        for it in b.items:
            if it["part"] == "PAD":
                nid = self.netid[b.padnet[(it["ref"], 1)]]
                for lay, r in ((L4, it["d"] / 2), (L3, TH_INNER)):
                    m = R2(it["x"], it["y"]) <= (r + G / 2) ** 2
                    self.static[lay][m] = nid
                    yy, xx = np.nonzero(m)
                    self.padcells.setdefault((it["ref"], 1), []).append((lay, yy, xx))
                self.drill_static |= R2(it["x"], it["y"]) <= (it["drill"] / 2 + 0.45 + VIA_D / 2) ** 2
                self.holes_static.append((it["x"], it["y"], it["drill"], True, b.padnet[(it["ref"], 1)]))
            elif it["part"] == "MOUNT":
                self.static[:, R2(it["x"], it["y"]) <= (it["drill"] / 2 + 0.30) ** 2] = -1
                self.drill_static |= R2(it["x"], it["y"]) <= (it["drill"] / 2 + 0.6) ** 2
                self.holes_static.append((it["x"], it["y"], it["drill"], False, None))
            elif it["side"] == "back":
                for p in b.pads_of(it):
                    m, c, a = self._raster_shape(p)
                    name = b.padnet.get((it["ref"], p["num"]))
                    yy, xx = np.nonzero(m)
                    self.static[L4][yy + c, xx + a] = self.netid[name] if name else -1
                    self.smd[yy + c, xx + a] = True
                    self.padcells.setdefault((it["ref"], p["num"]), []).append((L4, yy + c, xx + a))
                if it.get("ring"):
                    hx, hy = it["hole_xy"]; rg = it["ring"]
                    m = np.abs(np.sqrt(R2(hx, hy)) - rg["r"]) <= rg["w"] / 2 + G / 2
                    self.static[L4][m] = self.netid[b.padnet[(it["ref"], rg["pad"])]]
                    self.smd[m] = True
                    yy, xx = np.nonzero(m)
                    self.padcells[(it["ref"], rg["pad"])].append((L4, yy, xx))
                if it.get("hole"):
                    hx, hy = it["hole_xy"]
                    m = R2(hx, hy) <= (it["hole"] / 2 + 0.25) ** 2
                    self.static[L3][m] = -1
                    self.static[L4][m & (self.static[L4] == 0)] = -1
                    self.drill_static |= R2(hx, hy) <= (it["hole"] / 2 + 0.6) ** 2
                    self.holes_static.append((hx, hy, it["hole"], False, None))
        self.smd_keep = ndi.distance_transform_edt(~self.smd) * G < (VIA_PAD / 2 + 0.10)       # no via in or against a pad
        x0, x1, y0, y1 = ISLAND
        self.in_island = (self.X > x0 + 0.5) & (self.X < x1 - 0.5) & (self.Y > y0 + 0.5) & (self.Y < y1 - 0.5)
        self.out_island = (self.X < x0 - 0.5) | (self.X > x1 + 0.5) | (self.Y < y0 - 0.5) | (self.Y > y1 + 0.5)
        self.fixed_vias = []

    def preplace_via(self, x, y, name, plane):
        iy, ix = self.ij(x, y)
        nid = self.netid[name]
        for lay in (L4, L3):
            _stamp(self.static[lay], iy, ix, (VIA_PAD / 2) / G + 0.5, nid)
        _stamp(self.drill_static, iy, ix, VIA_PITCH / G, True)
        self.fixed_vias.append(dict(x=float(ix * G - self.h), y=float(iy * G - self.h), net=name, plane=plane))
        self._rebuild()

    # ----------------------------------------------------------------------------------- bookkeeping
    def _jid(self, job):
        if job not in self.jobid:
            self.jobid[job] = len(self.jobid) + 1
        return self.jobid[job]

    def _stamp_route(self, job):
        rec = self.routes[job]
        nid, jid = self.netid[rec["net"]], self._jid(job)
        for lay, iy, ix, w in rec["segs"]:
            for q in range(len(iy)):
                _stamp(self.owner[lay], iy[q], ix[q], (w / 2) / G + 0.5, nid)
                _stamp(self.jobmap[lay], iy[q], ix[q], (w / 2) / G + 0.5, jid)
        for iy, ix, plane in rec["vias"]:
            for lay in (L4, L3):
                _stamp(self.owner[lay], iy, ix, (VIA_PAD / 2) / G + 0.5, nid)
                _stamp(self.jobmap[lay], iy, ix, (VIA_PAD / 2) / G + 0.5, jid)
            _stamp(self.drill_keep, iy, ix, VIA_PITCH / G, True)
            _stamp(self.viajob, iy, ix, VIA_PITCH / G, jid)

    def _rebuild(self):
        self.owner = self.static.copy()
        self.jobmap = np.zeros((2, self.n, self.n), np.int32)
        self.viajob = np.zeros((self.n, self.n), np.int32)
        self.drill_keep = self.drill_static.copy()
        for job in self.routes:
            self._stamp_route(job)

    def _lockmask(self):
        if not self.locked:
            return np.zeros((2, self.n, self.n), bool)
        return np.isin(self.jobmap, [self.jobid[j] for j in self.locked if j in self.jobid])

    # ----------------------------------------------------------------------------------- search
    def _maps(self, nid, hw, soft, allow_vias):
        """block (hard), viaok, extra cost per cell, extra cost per via site."""
        n = self.n
        block = np.zeros((2, n, n), np.uint8)
        extra = self.hist.copy()
        vextra = np.zeros((n, n), np.float32)
        viaok = (~self.drill_static & ~self.smd_keep) if allow_vias else np.zeros((n, n), bool)
        lock = self._lockmask() if soft else None
        need_t, need_v = hw + CLR + MARGIN, VIA_PAD / 2 + CLR + MARGIN
        for lay in (L4, L3):
            if soft:
                hard = ((self.static[lay] != 0) & (self.static[lay] != nid)) | (lock[lay] & (self.owner[lay] != nid))
                dyn = (self.jobmap[lay] != 0) & (self.owner[lay] != nid) & ~lock[lay]
                dh = ndi.distance_transform_edt(~hard) * G
                block[lay] = dh < need_t
                if allow_vias:
                    viaok &= dh >= need_v
                if dyn.any():
                    dd = ndi.distance_transform_edt(~dyn) * G
                    extra[lay] += np.where(dd < need_t, self.PEN, 0.0).astype(np.float32)
                    vextra += np.where(dd < need_v, self.PEN, 0.0).astype(np.float32)
            else:
                foreign = (self.owner[lay] != 0) & (self.owner[lay] != nid)
                d = ndi.distance_transform_edt(~foreign) * G
                block[lay] = d < need_t
                if allow_vias:
                    viaok &= d >= need_v
        if allow_vias:
            if soft:
                vextra += np.where(self.drill_keep & ~self.drill_static, self.PEN, 0.0).astype(np.float32)
            else:
                viaok &= ~self.drill_keep
        return block, viaok, extra, vextra

    def _search(self, nid, w, src_mask, tgt_mask, soft=False, allow_vias=True, via_cost=40.0, lcost=(1.0, 0.9), via_region=None):
        block, viaok, extra, vextra = self._maps(nid, w / 2, soft, allow_vias)
        block[src_mask] = 0
        block[tgt_mask] = 0
        tgt = tgt_mask
        if via_region is not None:                            # plane stub: any legal via site inside the plane's region is a goal
            vt = viaok & via_region
            if soft:
                vt = vt & (vextra == 0)
            tgt = tgt_mask.copy()
            tgt[L4] |= vt & (block[L4] == 0)
            tgt[L3] |= vt & (block[L3] == 0)
        src = np.flatnonzero(src_mask.ravel()).astype(np.int32)
        ty, tx = np.nonzero(tgt.any(0))
        if len(src) == 0 or len(ty) == 0:
            return np.empty(0, np.int32), tgt
        path = _astar(block, viaok, src, tgt, self.n, np.array(lcost, np.float64), via_cost, int(tx.min()), int(tx.max()), int(ty.min()), int(ty.max()), extra, vextra)
        return path, tgt

    def _split(self, path, w):
        """Node path (start -> goal) -> segments per layer and the via nodes between them."""
        nn = self.n * self.n
        lay, iy, ix = path // nn, (path % nn) // self.n, (path % nn) % self.n
        segs, vias = [], []
        start = 0
        for k in range(1, len(path) + 1):
            if k == len(path) or lay[k] != lay[start]:
                if k - start > 1:
                    segs.append((int(lay[start]), iy[start:k].copy(), ix[start:k].copy(), w))
                if k < len(path):
                    vias.append((int(iy[k]), int(ix[k]), None))
                start = k
        return segs, vias

    def _cells(self, key, mask):
        for lay, yy, xx in self.padcells[key]:
            mask[lay, yy, xx] = True

    def _victims(self, rec, nid):
        """Routed jobs of other nets that the new copper comes too close to."""
        out = set()
        n = self.n
        for lay, iy, ix, w in rec["segs"]:
            m = np.zeros((n, n), bool); m[iy, ix] = True
            y0, y1, x0, x1 = max(iy.min() - 12, 0), min(iy.max() + 13, n), max(ix.min() - 12, 0), min(ix.max() + 13, n)
            near = ndi.distance_transform_edt(~m[y0:y1, x0:x1]) * G < (w / 2 + CLR + MARGIN)
            jm = self.jobmap[lay][y0:y1, x0:x1]
            ids = np.unique(jm[near & (self.owner[lay][y0:y1, x0:x1] != nid)])
            out |= set(int(i) for i in ids if i > 0)
        for iy, ix, _ in rec["vias"]:
            R = int(np.ceil((VIA_PAD / 2 + CLR + MARGIN) / G))
            yy, xx = np.mgrid[max(iy - R, 0):min(iy + R + 1, n), max(ix - R, 0):min(ix + R + 1, n)]
            near = (yy - iy) ** 2 + (xx - ix) ** 2 <= ((VIA_PAD / 2 + CLR + MARGIN) / G) ** 2
            for lay in (L4, L3):
                jm = self.jobmap[lay][yy, xx]
                ids = np.unique(jm[near & (self.owner[lay][yy, xx] != nid)])
                out |= set(int(i) for i in ids if i > 0)
            if self.viajob[iy, ix] > 0:
                out.add(int(self.viajob[iy, ix]))
        inv = {v: k for k, v in self.jobid.items()}
        return [inv[i] for i in out if inv[i] in self.routes and inv[i] not in self.locked]

    def _commit(self, job, rec, used_soft):
        """Store a route; if it was pushed through other nets, remove those and return them for re-routing."""
        nid = self.netid[rec["net"]]
        victims = self._victims(rec, nid) if used_soft else []
        victims = [v for v in victims if v != job]
        if victims:
            for lay, iy, ix, w in rec["segs"]:
                self.hist[lay, iy, ix] += 1.5
        for v in victims:
            self.rips[v] = self.rips.get(v, 0) + 1
            if self.rips[v] >= self.MAX_RIPS:
                self.locked.add(v)
            del self.routes[v]
            self.n_ripups += 1
        self.routes[job] = rec
        if victims:
            self._rebuild()
        else:
            self._stamp_route(job)
        return victims

    # ----------------------------------------------------------------------------------- nets
    def route_signal(self, job):
        _, name, w, allow_vias = job
        nid = self.netid[name]
        pads = list(self.b.nets[name])
        tree = np.zeros((2, self.n, self.n), bool)
        self._cells(pads[0], tree)
        todo = pads[1:]
        rec = dict(net=name, segs=[], vias=[])
        used_soft = False
        nn = self.n * self.n
        while todo:
            tgt = np.zeros_like(tree)
            for key in todo:
                self._cells(key, tgt)
            path = np.empty(0, np.int32)
            for soft, vias_ok in ((False, allow_vias), (False, True), (True, True)):
                path, _ = self._search(nid, w, tree, tgt, soft=soft, allow_vias=vias_ok)
                if len(path):
                    used_soft |= soft
                    break
            if len(path) == 0:
                self.failed.append((name, tuple(f"{r}.{q}" for r, q in todo)))
                return None
            segs, vias = self._split(path[::-1], w)
            rec["segs"] += segs; rec["vias"] += vias
            for lay, iy, ix, _ in segs:
                tree[lay, iy, ix] = True
            for iy, ix, _ in vias:
                tree[:, iy, ix] = True
            gl, gy, gx = path[0] // nn, (path[0] % nn) // self.n, (path[0] % nn) % self.n
            for key in [k for k in todo if any(l == gl and ((yy == gy) & (xx == gx)).any() for l, yy, xx in self.padcells[k])]:
                todo.remove(key)
                self._cells(key, tree)
        return self._commit(job, rec, used_soft)

    def _plane_conn(self, name):
        """Copper of a plane net that already reaches its plane."""
        nid = self.netid[name]
        seed = np.zeros((2, self.n, self.n), bool)
        for key in self.b.nets[name]:
            if key[0].startswith("T") and "." in key[0]:
                self._cells(key, seed)
        for v in self.fixed_vias:
            if v["net"] == name and v["plane"]:
                iy, ix = self.ij(v["x"], v["y"]); seed[:, iy, ix] = True
        for rec in self.routes.values():
            if rec["net"] == name:
                for iy, ix, plane in rec["vias"]:
                    if plane:
                        seed[:, iy, ix] = True
        if not seed.any():
            return seed
        out = np.zeros_like(seed)
        for _ in range(3):                                    # copper islands joined through vias: a few rounds are enough
            for lay in (L4, L3):
                lab, _n = ndi.label(self.owner[lay] == nid)
                ids = np.unique(lab[seed[lay] | out[lay]]); ids = ids[ids > 0]
                out[lay] = np.isin(lab, ids)
            for rec in self.routes.values():
                if rec["net"] == name:
                    for iy, ix, _p in rec["vias"]:
                        if out[L4, iy, ix] or out[L3, iy, ix]:
                            out[:, iy, ix] = True
        return out

    def route_plane(self, job):
        _, name, key, w = job
        nid = self.netid[name]
        region = {"GND": np.ones((self.n, self.n), bool), "+VDRIVE": self.out_island, "+3V3": self.in_island}[name]
        plane = {"GND": "L1", "+VDRIVE": "L2", "+3V3": "L2"}[name]
        conn = self._plane_conn(name)
        src = np.zeros_like(conn)
        self._cells(key, src)
        if (src & conn).any():
            return []
        path, used_soft, tgt = np.empty(0, np.int32), False, None
        for soft in (False, True):
            path, tgt = self._search(nid, w, src, conn, soft=soft, via_cost=25.0, via_region=region)
            if len(path):
                used_soft = soft
                break
        if len(path) == 0:
            self.failed.append((name, (f"{key[0]}.{key[1]}",)))
            return None
        nn = self.n * self.n
        gl, gy, gx = path[0] // nn, (path[0] % nn) // self.n, (path[0] % nn) % self.n
        segs, vias = self._split(path[::-1], w)
        vias = [(iy, ix, plane if region[iy, ix] else None) for iy, ix, _ in vias]
        if not conn[gl, gy, gx]:
            vias.append((int(gy), int(gx), plane))               # ended on a free via site: drill it
        return self._commit(job, dict(net=name, segs=segs, vias=vias), used_soft)

    # ----------------------------------------------------------------------------------- results
    def results(self):
        tracks, vias = [], [dict(v) for v in self.fixed_vias]
        for job, rec in self.routes.items():
            for lay, iy, ix, w in rec["segs"]:
                pts = [(float(ix[0] * G - self.h), float(iy[0] * G - self.h))]
                for q in range(1, len(iy) - 1):                 # corner points only
                    if (ix[q] - ix[q - 1], iy[q] - iy[q - 1]) != (ix[q + 1] - ix[q], iy[q + 1] - iy[q]):
                        pts.append((float(ix[q] * G - self.h), float(iy[q] * G - self.h)))
                pts.append((float(ix[-1] * G - self.h), float(iy[-1] * G - self.h)))
                tracks.append(dict(layer="L4" if lay == L4 else "L3", net=rec["net"], w=w, pts=pts))
            for iy, ix, plane in rec["vias"]:
                vias.append(dict(x=float(ix * G - self.h), y=float(iy * G - self.h), net=rec["net"], plane=plane))
        return tracks, vias


def _jobs(b):
    """Routing order: driver outputs (local), then the microcontroller's pins from the hardest to escape to the easiest,
    then everything else.  Later nets that get boxed in push earlier ones out of the way (rip-up)."""
    is_th = lambda key: key[0].startswith("T") and "." in key[0]
    u1 = b.item("U1")
    pads = {p["num"]: p for p in b.pads_of(u1)}

    def rank(num):                                            # 0: row pads over the escape channel, 1: other row pads, 2: column pads
        p = pads[num]
        dx, dy = p["x"] - u1["x"], p["y"] - u1["y"]
        if abs(dy) > abs(dx):
            return (0, abs(dx)) if abs(dx) <= 1.45 else (1, -abs(dx))
        return (2, abs(dy))

    width = lambda k: W_PWR if k in ("+1V1", "+5V") else W_OUT if k.startswith("OUT") else W_SIG
    sig = [k for k in b.nets if k not in PLANE_NETS]
    outs = sorted([k for k in sig if k.startswith("OUT")], key=lambda s: int(s[3:]))
    jobs = [("sig", k, W_OUT, False) for k in outs]
    seen = set(outs)
    for num in sorted((n for n in pads if n != 57), key=rank):
        name = b.padnet.get(("U1", num))
        if not name or name == "+1V1":
            continue
        if name in PLANE_NETS:
            jobs.append(("plane", name, ("U1", num), W_PWR))
        elif name not in seen:
            seen.add(name)
            jobs.append(("sig", name, width(name), True))
    jobs += [("sig", k, width(k), True) for k in sig if k not in seen]
    have = set(j[2] for j in jobs if j[0] == "plane")
    for name in PLANE_NETS:
        jobs += [("plane", name, key, 0.30 if name == "+VDRIVE" else W_PWR) for key in b.nets[name] if not is_th(key) and key not in have and key != ("U1", 57)]
    return jobs


def route(board=None, verbose=True, max_steps=1500):
    """Route every net; returns the Router (see .failed, .results())."""
    from collections import deque
    b = board or Board()
    r = Router(b, verbose)
    u1 = b.item("U1")
    for dx, dy in ((0, 0), (0.9, 0.9), (-0.9, 0.9), (0.9, -0.9), (-0.9, -0.9)):        # ground vias in the exposed pad of the RP2040
        r.preplace_via(u1["x"] + dx, u1["y"] + dy, "GND", "L1")
    all_jobs = _jobs(b)
    queue = deque(all_jobs)
    steps = 0
    tries = {}
    while queue and steps < max_steps:
        job = queue.popleft()
        steps += 1
        victims = r.route_signal(job) if job[0] == "sig" else r.route_plane(job)
        if victims is None and tries.get(job, 0) < 4:              # boxed in for now: try again after the others have moved
            tries[job] = tries.get(job, 0) + 1
            queue.append(job)
        if victims:
            for v in reversed(victims):
                if v in queue:
                    queue.remove(v)
                queue.appendleft(v)
        if verbose and steps % 60 == 0:
            print(f"   ... {steps} routing steps, {len(queue)} to go, {r.n_ripups} rip-ups, {len(r.failed)} failures so far", flush=True)
    # a ripped-up plane stub can leave another stub that leaned on it without its plane: repair until stable
    for _ in range(4):
        again = [j for j in all_jobs if j[0] == "plane"]
        before = len(r.routes)
        r.failed = [f for f in r.failed if f[0] not in PLANE_NETS]
        for j in again:
            if j in r.routes:
                continue
            v = r.route_plane(j)
            for q in v or []:
                (r.route_signal(q) if q[0] == "sig" else r.route_plane(q))
        if len(r.routes) == before:
            break
    done = set(r.routes)
    r.failed = [f for f in dict.fromkeys(r.failed) if any(j not in done for j in all_jobs if j[1] == f[0])] 
    r.missing = [j for j in all_jobs if j not in done and not (j[0] == "plane" and (np.any([m[l, yy, xx].any() for l, yy, xx in r.padcells[j[2]]]) if (m := r._plane_conn(j[1])) is not None else False))]
    r.steps = steps
    if verbose:
        print(f"  routing: {steps} steps, {r.n_ripups} rip-ups, {len(r.routes)} routes, not completed: {[(j[1], j[2] if j[0] == 'plane' else '') for j in r.missing]}", flush=True)
    return r


# =============================================================================================
# fabrication and assembly files
# =============================================================================================
class _Gerber:
    def __init__(self, function, off):
        self.ap, self.body, self.off, self.pol = {}, [], off, "D"
        self.head = ["%TF.GenerationSoftware,replicator,pcbroute.py,2*%", f"%TF.FileFunction,{function}*%", "%FSLAX46Y46*%", "%MOMM*%", "%LPD*%"]

    def _d(self, key):
        if key not in self.ap:
            self.ap[key] = 10 + len(self.ap)
        return self.ap[key]

    def _xy(self, x, y):
        return f"X{int(round((x + self.off) * 1e6))}Y{int(round((y + self.off) * 1e6))}"

    def polarity(self, dark):
        p = "D" if dark else "C"
        if p != self.pol:
            self.body.append(f"%LP{p}*%"); self.pol = p

    def flash(self, x, y, w, h=None, shape="C"):
        key = ("C", round(w, 4)) if shape == "C" or h is None else (shape, round(w, 4), round(h, 4))
        self.body.append(f"D{self._d(key)}*"); self.body.append(self._xy(x, y) + "D03*")

    def line(self, pts, width):
        self.body.append(f"D{self._d(('C', round(width, 4)))}*")
        self.body.append(self._xy(*pts[0]) + "D02*")
        for p in pts[1:]:
            self.body.append(self._xy(*p) + "D01*")

    def region(self, pts):
        self.body.append("G36*"); self.body.append(self._xy(*pts[0]) + "D02*")
        for p in list(pts[1:]) + [pts[0]]:
            self.body.append(self._xy(*p) + "D01*")
        self.body.append("G37*")

    def circle(self, x, y, r, width, n=48):
        self.line([(x + r * np.cos(2 * np.pi * i / n), y + r * np.sin(2 * np.pi * i / n)) for i in range(n + 1)], width)

    def pad(self, p, grow=0.0):
        if p.get("poly"):
            self.region(p["poly"])
        elif p["shape"] == "C":
            self.flash(p["x"], p["y"], p["w"] + grow)
        else:
            self.flash(p["x"], p["y"], p["w"] + grow, p["h"] + grow, "O" if p["shape"] == "O" else "R")

    def text(self):
        defs = [f"%ADD{n}C,{k[1]:.4f}*%" if k[0] == "C" else f"%ADD{n}{k[0]},{k[1]:.4f}X{k[2]:.4f}*%" for k, n in self.ap.items()]
        return "\n".join(self.head + defs + ["G01*"] + self.body + ["M02*"]) + "\n"


def _clean_plane(g, side, extra_seeds, min_web=0.14):
    """Remove from a plane everything a board house would complain about or that floats: copper webs thinner than
    min_web and fragments cut off from the plane by clearance holes.  Works on a rasterised copy of the layer and
    appends clear polygons to the Gerber.  The largest piece is kept, plus the piece containing each extra seed point."""
    import tempfile
    from skimage import measure, morphology
    with tempfile.NamedTemporaryFile("w", suffix=".gbr", delete=False) as f:
        f.write(g.text())
    img = _parse_gerber(f.name, side)
    os.remove(f.name)
    opened = morphology.binary_opening(img, morphology.disk(int(round(min_web / 2 / PX))))
    lab, n = ndi.label(opened)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = {int(np.argmax(sizes))}
    for x, y in extra_seeds:
        l = lab[int(round((y + side / 2) / PX)), int(round((x + side / 2) / PX))]
        if l:
            keep.add(int(l))
    kept = np.isin(lab, list(keep))
    # grow the kept copper back to its original edge (opening rounds corners), then everything else goes
    kept = morphology.binary_dilation(kept, morphology.disk(2)) & img
    remove = morphology.binary_dilation(img & ~kept, morphology.disk(1))
    rl, rn = ndi.label(remove)
    g.polarity(False)
    cnt = 0
    for sl in ndi.find_objects(rl):
        y0, x0 = max(sl[0].start - 2, 0), max(sl[1].start - 2, 0)
        sub = np.pad(rl[max(sl[0].start - 2, 0):sl[0].stop + 2, max(sl[1].start - 2, 0):sl[1].stop + 2] > 0, 1)
        for c in measure.find_contours(sub.astype(float), 0.5):
            if len(c) >= 4:
                pts = [((cc[1] - 1 + x0) * PX - side / 2, (cc[0] - 1 + y0) * PX - side / 2) for cc in c[::2]]
                if len(pts) >= 3:
                    g.region(pts); cnt += 1
    return cnt


TH_GAP = 0.30          # ground pour to a transducer pad that is not ground
VIA_GAP = 0.20         # ground pour / plane to a via pad or barrel of another net
ANTI_VIA = VIA_D / 2 + 0.35
ANTI_TH = 0.95


def write_fab(b: Board, tracks, vias, outdir, stem="TB-25"):
    """Gerber + drill files (names as JLCPCB's viewer detects them)."""
    os.makedirs(outdir, exist_ok=True)
    off = b.side / 2
    L1, L2, L3g, L4g = _Gerber("Copper,L1,Top", off), _Gerber("Copper,L2,Inr", off), _Gerber("Copper,L3,Inr", off), _Gerber("Copper,L4,Bot", off)
    fmask, bmask, bpaste = _Gerber("Soldermask,Top", off), _Gerber("Soldermask,Bot", off), _Gerber("Paste,Bot", off)
    fsilk, bsilk, edge = _Gerber("Legend,Top", off), _Gerber("Legend,Bot", off), _Gerber("Profile,NP", off)
    edge.line(b.outline(), 0.10)
    h, ch, e = b.side / 2 - EDGE, b.chamfer, EDGE
    inset = [(-h + ch, -h), (h - ch, -h), (h, -h + ch), (h, h - ch), (h - ch, h), (-h + ch, h), (-h, h - ch), (-h, -h + ch)]
    ths = [i for i in b.items if i["part"] == "PAD"]
    npth = [(i["x"], i["y"], i["drill"]) for i in b.items if i["part"] == "MOUNT"] + [(i["hole_xy"][0], i["hole_xy"][1], i["hole"]) for i in b.items if i.get("hole")]
    net_of_th = lambda i: b.padnet[(i["ref"], 1)]
    # ---- L1: ground pour
    L1.region(inset)
    L1.polarity(False)
    for i in ths:
        L1.flash(i["x"], i["y"], i["d"] + 2 * TH_GAP)
    for x, y, d in npth:
        L1.flash(x, y, d + 0.6)
    L1.polarity(True)
    for i in ths:
        L1.flash(i["x"], i["y"], i["d"])
        if net_of_th(i) == "GND":                                   # thermal relief: four spokes
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                L1.line([(i["x"] + dx * 0.8, i["y"] + dy * 0.8), (i["x"] + dx * (i["d"] / 2 + TH_GAP + 0.1), i["y"] + dy * (i["d"] / 2 + TH_GAP + 0.1))], 0.45)
    L1.polarity(False)
    for v in vias:
        if v["net"] != "GND":
            L1.flash(v["x"], v["y"], VIA_PAD + 2 * VIA_GAP)
    _clean_plane(L1, b.side, [])
    L1.polarity(True)
    for i in ths:
        L1.flash(i["x"], i["y"], i["d"])
    for v in vias:
        L1.flash(v["x"], v["y"], VIA_PAD)
    # ---- L2: +V_DRIVE plane with the +3V3 island
    L2.region(inset)
    L2.polarity(False)
    x0, x1, y0, y1 = ISLAND
    L2.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], 0.30)
    for i in ths:
        L2.flash(i["x"], i["y"], 2 * ANTI_TH)
    for x, y, d in npth:
        L2.flash(x, y, d + 0.6)
    for v in vias:
        if v["plane"] != "L2":
            L2.flash(v["x"], v["y"], 2 * ANTI_VIA)
    _clean_plane(L2, b.side, [((x0 + x1) / 2, (y0 + y1) / 2)])
    L2.polarity(True)
    # ---- L3 and L4
    for i in ths:
        L3g.flash(i["x"], i["y"], 2 * TH_INNER)
        L4g.flash(i["x"], i["y"], i["d"])
    for v in vias:
        L3g.flash(v["x"], v["y"], VIA_PAD); L4g.flash(v["x"], v["y"], VIA_PAD)
    for t in tracks:
        (L4g if t["layer"] == "L4" else L3g).line(t["pts"], t["w"])
    # ---- pads, mask, paste, legend
    for i in ths:
        fmask.flash(i["x"], i["y"], i["d"] + 0.1); bmask.flash(i["x"], i["y"], i["d"] + 0.1)
    for x, y, d in npth:
        fmask.flash(x, y, d + 0.3); bmask.flash(x, y, d + 0.3)
    for it in b.items:
        if it["part"] == "TRANSDUCER":
            fsilk.circle(it["x"], it["y"], it["d"] / 2, 0.15)
            for poly in b.text(it["ref"], it["x"], it["y"] + 2.6, 1.3, False):
                fsilk.line(poly, 0.15)
            fsilk.line([(it["x"] - 3.7, it["y"] - 0.6), (it["x"] - 3.7, it["y"] + 0.6)], 0.15)       # the driven lead is on this side
            continue
        if it["side"] != "back":
            continue
        fp = FOOTPRINTS[it["part"]]
        if it.get("ring"):
            L4g.circle(it["hole_xy"][0], it["hole_xy"][1], it["ring"]["r"], it["ring"]["w"])
        for p in b.pads_of(it):
            L4g.pad(p)
            fine = min(p["w"], p["h"]) <= 0.25
            if it["part"] == "FID":
                bmask.flash(p["x"], p["y"], 2.0)
            else:
                bmask.pad(p, 0.05 if fine else 0.10)
            if it["part"] in ("TP", "FID") or it.get("value") == "120R":
                continue
            if it["part"] == "RP2040" and p["num"] == 57:                # exposed pad: paste windows between the ground vias
                for dx, dy in ((0.9, 0), (-0.9, 0), (0, 0.9), (0, -0.9)):
                    bpaste.flash(p["x"] + dx, p["y"] + dy, 1.0, 1.0, "R")
            else:
                bpaste.pad(p)
        if it["part"] in ("TP", "FID"):
            continue
        bw, bh, _ = fp["body"]
        if it["rot"] in (90, 270):
            bw, bh = bh, bw
        p1 = next(q for q in b.pads_of(it) if q["num"] == 1)
        big = it["part"] in ("RP2040", "DRV", "RS485", "FLASH", "LDO", "J_BUS", "J_PWR", "J_SWD", "MIC", "XTAL")
        if big:
            xa, xb, ya, yb = it["x"] - bw / 2, it["x"] + bw / 2, it["y"] - bh / 2, it["y"] + bh / 2
            t = 0.6
            for (cx, cy, sx, sy) in ((xa, ya, 1, 1), (xb, ya, -1, 1), (xb, yb, -1, -1), (xa, yb, 1, -1)):       # corner ticks: the pads fill the sides
                bsilk.line([(cx + sx * t, cy), (cx, cy), (cx, cy + sy * t)], 0.12)
            if it["part"] not in ("MIC", "XTAL"):
                for poly in b.text(it["ref"], it["x"], it["y"], 0.9, True):
                    bsilk.line(poly, 0.12)
        if it["part"] in ("RP2040", "DRV", "RS485", "FLASH", "LDO", "MIC", "XTAL", "SOT23", "LED0603", "J_BUS", "J_PWR", "J_SWD"):
            # pin-1 dot just outside pad 1, away from the part centre
            v = np.array([p1["x"] - it["x"], p1["y"] - it["y"]])
            ax = 0 if abs(v[0]) * (1 if p1["w"] >= p1["h"] else 0.5) > abs(v[1]) * (1 if p1["h"] > p1["w"] else 0.5) else 1
            dpos = np.array([p1["x"], p1["y"]])
            dpos[ax] += np.sign(v[ax]) * ((p1["w"] if ax == 0 else p1["h"]) / 2 + 0.35)
            bsilk.flash(dpos[0], dpos[1], 0.3)
    for poly in b.text("TB-25 REV C", 0.0, -26.6, 1.0, True):
        bsilk.line(poly, 0.13)
    files = {}
    for name, g in (("F_Cu.gtl", L1), ("In1_Cu.g2", L2), ("In2_Cu.g3", L3g), ("B_Cu.gbl", L4g), ("F_Mask.gts", fmask), ("B_Mask.gbs", bmask), ("B_Paste.gbp", bpaste),
                    ("F_Silkscreen.gto", fsilk), ("B_Silkscreen.gbo", bsilk), ("Edge_Cuts.gm1", edge)):
        files[f"{stem}-{name}"] = g.text()
    pth = {VIA_D: [(v["x"], v["y"]) for v in vias], TRANSDUCER["drill"]: [(i["x"], i["y"]) for i in ths]}
    np_ = {}
    for x, y, d in npth:
        np_.setdefault(round(d, 3), []).append((x, y))
    for name, tools, plated in (("PTH.drl", pth, True), ("NPTH.drl", np_, False)):
        o = ["M48", f"; {'plated' if plated else 'non-plated'} holes, {stem}", "METRIC,TZ", "FMAT,2"]
        keys = sorted(tools)
        o += [f"T{i+1}C{d:.3f}" for i, d in enumerate(keys)] + ["%", "G90", "G05"]
        for i, d in enumerate(keys):
            o.append(f"T{i+1}")
            o += [f"X{x + off:.3f}Y{y + off:.3f}" for x, y in tools[d]]
        o += ["T0", "M30"]
        files[f"{stem}-{name}"] = "\n".join(o) + "\n"
    for fn in os.listdir(outdir):
        if fn.startswith(stem + "-"):
            os.remove(os.path.join(outdir, fn))
    for fn, txt in files.items():
        open(os.path.join(outdir, fn), "w").write(txt)
    return sorted(files)


def write_assembly(b: Board, outdir, stem="TB-25"):
    """JLCPCB bill of materials and pick-and-place files for the back side."""
    groups, cpl = {}, ["Designator,Mid X,Mid Y,Layer,Rotation"]
    off = b.side / 2
    for it in b.items:
        if it["side"] != "back" or it["part"] in ("TP", "FID"):
            continue
        key = it.get("value") or it["part"]
        desc, lcsc, cls = ORDER[key]
        if key == "120R":
            continue                                             # fitted by hand on the last board of each chain
        groups.setdefault((desc, FOOTPRINTS[it["part"]]["lib_package"], lcsc, cls), []).append(it["ref"])
        cpl.append(f"{it['ref']},{it['x'] + off:.3f}mm,{it['y'] + off:.3f}mm,Bottom,{it['rot']}")
    bom = ["Comment,Designator,Footprint,LCSC Part #"]
    for (desc, pkg, lcsc, cls), refs in sorted(groups.items(), key=lambda kv: kv[1][0]):
        bom.append(f"\"{desc}\",\"{','.join(refs)}\",\"{pkg}\",{lcsc}")
    open(os.path.join(outdir, f"{stem}_BOM_JLCPCB.csv"), "w").write("\n".join(bom) + "\n")
    open(os.path.join(outdir, f"{stem}_CPL_JLCPCB.csv"), "w").write("\n".join(cpl) + "\n")
    return dict(bom_lines=len(bom) - 1, placed=len(cpl) - 1, extended=sorted(set(k[2] for k in groups if k[3] == "extended")),
                groups=[dict(desc=k[0], package=k[1], lcsc=k[2], cls=k[3], qty=len(v), refs=v) for k, v in groups.items()])


# =============================================================================================
# verification from the files alone
# =============================================================================================
PX = 0.02


def _parse_gerber(path, side):
    """Rasterise one of this module's Gerber files at PX mm.  Returns a boolean image (y up)."""
    from skimage.draw import polygon as fill_poly
    n = int(round(side / PX)) + 1
    img = np.zeros((n, n), bool)
    ap, cur, dark = {}, None, True
    x = y = 0.0
    region, contour = False, []
    yy, xx = np.mgrid[0:n, 0:n]

    def disc(cx, cy, r, val):
        a, b_ = max(int((cx - r) / PX) - 1, 0), min(int((cx + r) / PX) + 2, n)
        c, d = max(int((cy - r) / PX) - 1, 0), min(int((cy + r) / PX) + 2, n)
        m = (xx[c:d, a:b_] * PX - cx) ** 2 + (yy[c:d, a:b_] * PX - cy) ** 2 <= r * r
        img[c:d, a:b_][m] = val

    def rect(cx, cy, w, h, val):
        a, b_ = max(int(np.ceil((cx - w / 2) / PX - 1e-6)), 0), min(int(np.floor((cx + w / 2) / PX + 1e-6)) + 1, n)
        c, d = max(int(np.ceil((cy - h / 2) / PX - 1e-6)), 0), min(int(np.floor((cy + h / 2) / PX + 1e-6)) + 1, n)
        img[c:d, a:b_] = val

    def seg(x0, y0, x1, y1, r, val):
        a, b_ = max(int((min(x0, x1) - r) / PX) - 1, 0), min(int((max(x0, x1) + r) / PX) + 2, n)
        c, d = max(int((min(y0, y1) - r) / PX) - 1, 0), min(int((max(y0, y1) + r) / PX) + 2, n)
        X, Y = xx[c:d, a:b_] * PX, yy[c:d, a:b_] * PX
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        m = (X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2 <= r * r
        img[c:d, a:b_][m] = val

    for raw in open(path).read().replace("\n", "").split("*"):
        s = raw.strip().strip("%")
        if not s:
            continue
        if s.startswith("ADD"):
            num = int(s[3:5]); kind = s[5]; vals = [float(v) for v in s[7:].split("X")]
            ap[num] = (kind, vals)
        elif s in ("LPD", "LPC"):
            dark = s == "LPD"
        elif s == "G36":
            region, contour = True, []
        elif s == "G37":
            if len(contour) >= 3:
                c = np.array(contour) / PX
                r_, c_ = fill_poly(c[:, 1], c[:, 0], img.shape)
                img[r_, c_] = dark
            region = False
        elif s.startswith("D") and s[1:].isdigit() and int(s[1:]) >= 10:
            cur = ap[int(s[1:])]
        elif s.startswith("X"):
            ix, iy, d = s.index("Y"), s.index("D"), None
            nx, ny, op = int(s[1:ix]) / 1e6, int(s[ix + 1:iy]) / 1e6, s[iy:]
            if region:
                if op == "D02":
                    contour = [(nx, ny)]
                else:
                    contour.append((nx, ny))
            elif op == "D03":
                kind, v = cur
                if kind == "C":
                    disc(nx, ny, v[0] / 2, dark)
                elif kind == "R":
                    rect(nx, ny, v[0], v[1], dark)
                else:
                    r = min(v) / 2
                    hx, hy = max(v[0] / 2 - r, 0), max(v[1] / 2 - r, 0)
                    seg(nx - hx, ny - hy, nx + hx, ny + hy, r, dark)
            elif op == "D01":
                seg(x, y, nx, ny, cur[1][0] / 2, dark)
            x, y = nx, ny
    return img


def _parse_drill(path):
    tools, cur, out = {}, None, []
    for ln in open(path).read().splitlines():
        ln = ln.strip()
        if ln.startswith("T") and "C" in ln:
            tools[ln[:ln.index("C")]] = float(ln[ln.index("C") + 1:])
        elif ln.startswith("T") and ln in tools:
            cur = tools[ln]
        elif ln.startswith("X") and cur is not None:
            out.append((float(ln[1:ln.index("Y")]), float(ln[ln.index("Y") + 1:]), cur))
    return out


def verify(b: Board, outdir, stem="TB-25", rule=0.127):
    """Rebuild the connectivity from the Gerber and drill files and compare it with the netlist;
    measure copper clearances.  Uses nothing from the router."""
    off = b.side / 2
    layers = {k: _parse_gerber(os.path.join(outdir, f"{stem}-{fn}"), b.side) for k, fn in (("L1", "F_Cu.gtl"), ("L2", "In1_Cu.g2"), ("L3", "In2_Cu.g3"), ("L4", "B_Cu.gbl"))}
    pth = _parse_drill(os.path.join(outdir, f"{stem}-PTH.drl"))
    npth = _parse_drill(os.path.join(outdir, f"{stem}-NPTH.drl"))
    labs, base = {}, {}
    total = 0
    for k, img in layers.items():
        lab, cnt = ndi.label(img)
        labs[k] = lab; base[k] = total; total += cnt + 1
    parent = np.arange(total)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b_):
        ra, rb = find(a), find(b_)
        if ra != rb:
            parent[rb] = ra

    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    floating_holes = 0
    for x, y, d in pth:
        ids = []
        for k in layers:
            px = np.round((x + (d / 2 + PX) * np.cos(ang)) / PX).astype(int)
            py = np.round((y + (d / 2 + PX) * np.sin(ang)) / PX).astype(int)
            l = labs[k][py, px]
            ids += [base[k] + int(v) for v in np.unique(l[l > 0])]
        for a in ids[1:]:
            union(ids[0], a)
        floating_holes += len(ids) == 0
    # pads -> component
    net_roots, root_nets, lost = {}, {}, []
    for it in b.items:
        if it["part"] == "PAD":
            pads = [dict(num=1, x=it["x"] + it["drill"] / 2 + 0.2, y=it["y"])]
        elif it["side"] == "back":
            pads = b.pads_of(it)
        else:
            continue
        for p in pads:
            name = b.padnet.get((it["ref"], p["num"]))
            cx, cy = (np.mean(np.array(p["poly"]), 0) if p.get("poly") else (p["x"], p["y"]))
            iy, ix = int(round((cy + off) / PX)), int(round((cx + off) / PX))
            l = labs["L4"][iy, ix]
            if l == 0:
                win = labs["L4"][iy - 6:iy + 7, ix - 6:ix + 7]
                l = int(win[win > 0][0]) if (win > 0).any() else 0
            if l == 0:
                lost.append(f"{it['ref']}.{p['num']}")
                continue
            r = find(base["L4"] + int(l))
            key = name or f"(no net) {it['ref']}.{p['num']}"
            net_roots.setdefault(key, {}).setdefault(r, []).append(f"{it['ref']}.{p['num']}")
            root_nets.setdefault(r, set()).add(key)
    opens = {k: [v2 for v2 in v.values()] for k, v in net_roots.items() if len(v) > 1}
    shorts = [sorted(v) for v in root_nets.values() if len(v) > 1]
    # clearances between copper of different nets, per layer
    clear = {}
    worst = []
    for k, img in layers.items():
        lab = labs[k]
        roots = np.array([0] + [find(base[k] + i) for i in range(1, lab.max() + 1)])
        dist, idx = ndi.distance_transform_edt(~img, return_indices=True)
        near = roots[lab[idx[0], idx[1]]]
        near[img] = 0
        best = np.inf; where = None
        for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))), ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
            a, c = near[sl_a], near[sl_b]
            m = (a != c) & (a != 0) & (c != 0)
            if m.any():
                gsum = (dist[sl_a] + dist[sl_b])[m]
                j = int(np.argmin(gsum))
                if gsum[j] < best:
                    best = float(gsum[j]); yy, xx = np.nonzero(m); where = (float(xx[j] * PX - off), float(yy[j] * PX - off))
                bad = gsum * PX < rule
                if bad.any():
                    yy, xx = np.nonzero(m)
                    for q in np.flatnonzero(bad)[:5]:
                        worst.append(dict(layer=k, gap_mm=round(float(gsum[q] * PX), 3), at=[round(float(xx[q] * PX - off), 2), round(float(yy[q] * PX - off), 2)]))
        clear[k] = dict(min_gap_mm=round(best * PX, 3) if np.isfinite(best) else None, at=where)
        # copper to board edge
        ys, xs = np.nonzero(img)
        X, Y = np.abs(xs * PX - off), np.abs(ys * PX - off)
        clear[k]["edge_mm"] = round(float(min((off - X).min(), (off - Y).min(), ((2 * off - b.chamfer - X - Y) / np.sqrt(2)).min())), 3)
        clear[k]["copper_pct"] = round(100 * float(img.mean()), 1)
    # drilled holes: spacing wall to wall, and plated holes against foreign copper
    H = np.array([(x, y, d) for x, y, d in pth + npth])
    dd = np.hypot(H[:, None, 0] - H[None, :, 0], H[:, None, 1] - H[None, :, 1]) - (H[:, None, 2] + H[None, :, 2]) / 2
    np.fill_diagonal(dd, np.inf)
    ann = []
    for x, y, d in pth:                                             # annular ring on the outer layers
        for k in ("L1", "L4"):
            px = np.round((x + (d / 2 + 0.12) * np.cos(ang)) / PX).astype(int); py = np.round((y + (d / 2 + 0.12) * np.sin(ang)) / PX).astype(int)
            if not layers[k][py, px].all():
                ann.append((round(x - off, 2), round(y - off, 2), k))
    res = dict(opens=opens, shorts=shorts, pads_without_copper=lost, plated_holes=len(pth), non_plated_holes=len(npth), holes_touching_no_copper=floating_holes,
               clearance=clear, clearance_rule_mm=rule, clearance_violations=worst[:20], n_clearance_violations=len(worst),
               hole_wall_to_wall_min_mm=round(float(dd.min()), 3), annular_ring_under_0p12=ann[:10], n_annular=len(ann),
               nets_checked=len(net_roots), pads_checked=sum(len(x) for v in net_roots.values() for x in v.values()))
    res["ok"] = not opens and not shorts and not lost and not worst and floating_holes == 0 and not ann and res["hole_wall_to_wall_min_mm"] >= 0.25
    return res


def build(outdir, reroute=False, verbose=True):
    """Place, route (or reuse the stored routing if the placement is unchanged), write all files, verify them."""
    import hashlib
    b = Board()
    os.makedirs(outdir, exist_ok=True)
    key = hashlib.sha1(json.dumps([[i["ref"], i["part"], round(i["x"], 3), round(i["y"], 3), i["rot"]] for i in b.items] + sorted(map(str, b.padnet.items())) +
                                  [G, CLR, MARGIN, W_SIG, W_PWR, W_OUT, VIA_PAD, VIA_PITCH, list(ISLAND), "rev2"]).encode()).hexdigest()
    cache = os.path.join(outdir, "TB-25_routes.json")
    data = json.load(open(cache)) if os.path.exists(cache) else None
    if data and data.get("key") == key and not reroute:
        tracks, vias, info = data["tracks"], data["vias"], data["info"]
    else:
        r = route(b, verbose=verbose)
        tracks, vias = r.results()
        info = dict(steps=r.steps, ripups=r.n_ripups, not_completed=[[j[1], list(j[2]) if j[0] == "plane" else None] for j in r.missing])
        json.dump(dict(key=key, tracks=tracks, vias=vias, info=info), open(cache, "w"))
    gdir = os.path.join(outdir, "gerbers")
    files = write_fab(b, tracks, vias, gdir)
    asm = write_assembly(b, outdir)
    ver = verify(b, gdir)
    length = {}
    for t in tracks:
        p = np.array(t["pts"])
        length[t["net"]] = length.get(t["net"], 0.0) + float(np.hypot(*np.diff(p, axis=0).T).sum())
    rep = dict(routing=dict(info, tracks=len(tracks), vias=len(vias), plane_vias=sum(1 for v in vias if v["plane"]), total_track_mm=round(sum(length.values()), 1),
                            longest=sorted(((round(v, 1), k) for k, v in length.items()), reverse=True)[:5],
                            rules=dict(grid_mm=G, clearance_mm=CLR, track_signal_mm=W_SIG, track_power_mm=W_PWR, track_output_mm=W_OUT, via_drill_mm=VIA_D, via_pad_mm=VIA_PAD,
                                       stack="L1 ground pour / L2 +V_DRIVE plane with +3V3 island / L3 signals / L4 parts and signals")),
               files=files, assembly=asm, verification=ver)
    b.to_json(os.path.join(outdir, "TB-25.json"), extra=dict(tracks=tracks, vias=vias, report=rep))
    json.dump(rep, open(os.path.join(outdir, "route_report.json"), "w"), indent=1)
    b.to_svg(os.path.join(outdir, "TB-25_back.svg"), "back", tracks=tracks, vias=vias)
    b.to_svg(os.path.join(outdir, "TB-25_front.svg"), "front")
    return b, rep


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "hardware/pcb"
    b, rep = build(out, reroute="--reroute" in sys.argv)
    v = rep["verification"]
    print(f"TB-25 routed: {rep['routing']['tracks']} tracks ({rep['routing']['total_track_mm']} mm), {rep['routing']['vias']} vias, not completed: {rep['routing']['not_completed']}")
    print(f"  files: {len(rep['files'])} Gerber/drill; assembly: {rep['assembly']['placed']} parts placed, {rep['assembly']['bom_lines']} BOM lines, extended parts {rep['assembly']['extended']}")
    print(f"  check from the files: {v['nets_checked']} nets, {v['pads_checked']} pads; opens {len(v['opens'])}, shorts {len(v['shorts'])}, pads without copper {len(v['pads_without_copper'])}, "
          f"clearance violations {v['n_clearance_violations']}, annular {v['n_annular']}, hole spacing min {v['hole_wall_to_wall_min_mm']} mm")
    for k, c in v["clearance"].items():
        print(f"    {k}: smallest gap between different nets {c['min_gap_mm']} mm at {c['at']}, copper to edge {c['edge_mm']} mm, copper {c['copper_pct']} %")
    for k, o in list(v["opens"].items())[:12]:
        print("    OPEN", k, o)
    for s_ in v["shorts"][:12]:
        print("    SHORT", s_)
    for w in v["clearance_violations"][:12]:
        print("    GAP", w)
    print("  RESULT:", "PASS" if v["ok"] else "FAIL")


def selftest(b: Board, tracks, vias, tmpdir):
    """Does `verify` notice faults?  Three are injected into copies of the files: a deleted track (open circuit),
    a track bridging two nets (short circuit), and a track 0.06 mm from another net's pad (clearance)."""
    out = {}
    base = verify(b, write_fab(b, tracks, vias, tmpdir) and tmpdir)
    out["clean_passes"] = bool(base["ok"])
    # 1. delete the longest signal track
    k = max(range(len(tracks)), key=lambda i: (tracks[i]["net"].startswith("PH"), np.hypot(*np.diff(np.array(tracks[i]["pts"]), axis=0).T).sum()))
    write_fab(b, tracks[:k] + tracks[k + 1:], vias, tmpdir)
    r = verify(b, tmpdir)
    out["open_found"] = bool(tracks[k]["net"] in r["opens"])
    # 2. bridge two neighbouring driver pads of different nets
    p_a, p_b = b.pad("D1", 2), b.pad("D1", 3)
    write_fab(b, tracks + [dict(layer="L4", net="x", w=0.15, pts=[(p_a["x"], p_a["y"]), (p_b["x"], p_b["y"])])], vias, tmpdir)
    r = verify(b, tmpdir)
    na, nb = b.padnet[("D1", 2)], b.padnet[("D1", 3)]
    out["short_found"] = any(na in s and nb in s for s in r["shorts"])
    # 3. a stub of one net that stops 0.06 mm short of a pad of another net
    gap = 0.06
    xe = p_b["x"] + (p_b["w"] / 2 + gap + 0.075) * np.sign(p_a["x"] - p_b["x"])
    write_fab(b, tracks + [dict(layer="L4", net=na, w=0.15, pts=[(p_a["x"], p_a["y"]), (xe, p_a["y"])])], vias, tmpdir)
    r = verify(b, tmpdir)
    out["clearance_found"] = r["n_clearance_violations"] > 0 and not r["shorts"]
    out["smallest_gap_reported_mm"] = min((w["gap_mm"] for w in r["clearance_violations"]), default=None)
    return out
