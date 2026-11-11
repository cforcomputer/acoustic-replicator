"""Numba kernels: piston-array propagators and Helmholtz double-layer operators.

Conventions
-----------
Time dependence exp(-i w t); free-space Green function G = exp(ikr) / (4 pi r).
Surface normals point out of the solid, into the air.
"""
import math
import numpy as np
from numba import njit, prange

FOURPI = 4.0 * math.pi

# 3-point (degree 2) and 7-point (degree 5) Gauss rules on the reference triangle,
# given as barycentric coordinates + weights (weights sum to 1).
_Q3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
_W3 = np.array([1 / 3, 1 / 3, 1 / 3])
_a1, _b1 = 0.0597158717, 0.4701420641
_a2, _b2 = 0.7974269853, 0.1012865073
_Q7 = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
    [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2],
])
_W7 = np.array([0.225, 0.1323941527, 0.1323941527, 0.1323941527,
                0.1259391805, 0.1259391805, 0.1259391805])


@njit(parallel=True, cache=True)
def piston_rows(pts, sub, alpha, k, pref, want_grad):
    """Pressure (and gradient) at `pts` per unit complex drive of each transducer.

    Each transducer is a baffled piston discretised into S monopoles (Rayleigh
    integral):  p_t(x) = pref * sum_s alpha_s exp(ik r)/r,  sum(alpha) = 1,
    so the on-axis far field is pref * exp(ikd)/d.

    pts (n,3), sub (T,S,3), alpha (S,) -> A (n,T), G (n,3,T)
    """
    n = pts.shape[0]
    T = sub.shape[0]
    S = sub.shape[1]
    A = np.zeros((n, T), np.complex128)
    if want_grad:
        G = np.zeros((n, 3, T), np.complex128)
    else:
        G = np.zeros((1, 3, 1), np.complex128)
    for t in prange(T):
        for i in range(n):
            px = pts[i, 0]; py = pts[i, 1]; pz = pts[i, 2]
            acc = 0j; gx = 0j; gy = 0j; gz = 0j
            for s in range(S):
                dx = px - sub[t, s, 0]
                dy = py - sub[t, s, 1]
                dz = pz - sub[t, s, 2]
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                e = complex(math.cos(k * r), math.sin(k * r)) * (alpha[s] / r)
                acc += e
                if want_grad:
                    f = e * complex(-1.0, k * r) / (r * r)   # d/dr(e^{ikr}/r) / r
                    gx += f * dx; gy += f * dy; gz += f * dz
            A[i, t] = pref * acc
            if want_grad:
                G[i, 0, t] = pref * gx
                G[i, 1, t] = pref * gy
                G[i, 2, t] = pref * gz
    return A, G


@njit(cache=True)
def _solid_angle(x, a, b, c):
    """Signed solid angle of triangle (a,b,c) seen from x (van Oosterom-Strackee).
    Positive when the triangle normal (b-a)x(c-a) points away from x."""
    r1 = a - x; r2 = b - x; r3 = c - x
    n1 = math.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2)
    n2 = math.sqrt(r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2)
    n3 = math.sqrt(r3[0] ** 2 + r3[1] ** 2 + r3[2] ** 2)
    cx = r2[1] * r3[2] - r2[2] * r3[1]
    cy = r2[2] * r3[0] - r2[0] * r3[2]
    cz = r2[0] * r3[1] - r2[1] * r3[0]
    num = r1[0] * cx + r1[1] * cy + r1[2] * cz
    d12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    d13 = r1[0] * r3[0] + r1[1] * r3[1] + r1[2] * r3[2]
    d23 = r2[0] * r3[0] + r2[1] * r3[1] + r2[2] * r3[2]
    den = n1 * n2 * n3 + d12 * n3 + d13 * n2 + d23 * n1
    return 2.0 * math.atan2(num, den)


@njit(cache=True)
def _seg_biot(x, a, b, out):
    """Add  int_a^b dl x (x-y)/|x-y|^3  to out (3,)."""
    r1 = x - a; r2 = x - b
    cx = r1[1] * r2[2] - r1[2] * r2[1]
    cy = r1[2] * r2[0] - r1[0] * r2[2]
    cz = r1[0] * r2[1] - r1[1] * r2[0]
    c2 = cx * cx + cy * cy + cz * cz
    n1 = math.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2)
    n2 = math.sqrt(r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2)
    if c2 < 1e-30 * (n1 * n2) ** 2 + 1e-300:
        return
    r0 = b - a
    s = (r0[0] * (r1[0] / n1 - r2[0] / n2) + r0[1] * (r1[1] / n1 - r2[1] / n2)
         + r0[2] * (r1[2] / n1 - r2[2] / n2)) / c2
    out[0] += cx * s; out[1] += cy * s; out[2] += cz * s


@njit(parallel=True, cache=True)
def dl_matrices(pts, tri, nrm, area, size, k, near_factor, self_idx, want_grad):
    """Double-layer operator rows for constant triangular panels.

    Dp[i,j] = int_{S_j} dG/dn_y (x_i, y) dS(y)
    Dg[i,:,j] = grad_x of the same.

    Near panels (centroid distance < near_factor*size_j) use the exact
    solid-angle expression for the singular static part plus 7-point quadrature
    of the smooth remainder; far panels use 3-point quadrature of the full kernel.
    self_idx[i] = j marks x_i as the centroid of flat panel j (kernel == 0).
    """
    n = pts.shape[0]
    N = tri.shape[0]
    Dp = np.zeros((n, N), np.complex128)
    if want_grad:
        Dg = np.zeros((n, 3, N), np.complex128)
    else:
        Dg = np.zeros((1, 3, 1), np.complex128)
    for j in prange(N):
        a = tri[j, 0]; b = tri[j, 1]; c = tri[j, 2]
        nj = nrm[j]
        cen = (a + b + c) / 3.0
        thresh2 = (near_factor * size[j]) ** 2
        grad_s = np.zeros(3)
        for i in range(n):
            if self_idx[i] == j:
                continue
            x = pts[i]
            d0 = cen[0] - x[0]; d1 = cen[1] - x[1]; d2 = cen[2] - x[2]
            near = (d0 * d0 + d1 * d1 + d2 * d2) < thresh2
            accp = 0j; g0 = 0j; g1 = 0j; g2 = 0j
            if near:
                om = _solid_angle(x, a, b, c)
                accp += -om / FOURPI
                if want_grad:
                    grad_s[0] = 0.0; grad_s[1] = 0.0; grad_s[2] = 0.0
                    _seg_biot(x, a, b, grad_s)
                    _seg_biot(x, b, c, grad_s)
                    _seg_biot(x, c, a, grad_s)
                    g0 += -grad_s[0] / FOURPI
                    g1 += -grad_s[1] / FOURPI
                    g2 += -grad_s[2] / FOURPI
                for q in range(7):
                    y0 = _Q7[q, 0] * a[0] + _Q7[q, 1] * b[0] + _Q7[q, 2] * c[0]
                    y1 = _Q7[q, 0] * a[1] + _Q7[q, 1] * b[1] + _Q7[q, 2] * c[1]
                    y2 = _Q7[q, 0] * a[2] + _Q7[q, 1] * b[2] + _Q7[q, 2] * c[2]
                    R0 = y0 - x[0]; R1 = y1 - x[1]; R2 = y2 - x[2]
                    r2 = R0 * R0 + R1 * R1 + R2 * R2
                    r = math.sqrt(r2)
                    Rn = R0 * nj[0] + R1 * nj[1] + R2 * nj[2]
                    kr = k * r
                    E = complex(math.cos(kr), math.sin(kr))
                    w = _W7[q] * area[j] / FOURPI
                    # remainder kernel: [E(ikr-1)+1]/r^3 * (R.n)
                    gr = (E * complex(-1.0, kr) + 1.0) / (r2 * r)
                    accp += w * gr * Rn
                    if want_grad:
                        h = -(E * complex(3.0 - kr * kr, -3.0 * kr) - 3.0) / (r2 * r2 * r) * Rn
                        g0 += w * (h * R0 - gr * nj[0])
                        g1 += w * (h * R1 - gr * nj[1])
                        g2 += w * (h * R2 - gr * nj[2])
            else:
                for q in range(3):
                    y0 = _Q3[q, 0] * a[0] + _Q3[q, 1] * b[0] + _Q3[q, 2] * c[0]
                    y1 = _Q3[q, 0] * a[1] + _Q3[q, 1] * b[1] + _Q3[q, 2] * c[1]
                    y2 = _Q3[q, 0] * a[2] + _Q3[q, 1] * b[2] + _Q3[q, 2] * c[2]
                    R0 = y0 - x[0]; R1 = y1 - x[1]; R2 = y2 - x[2]
                    r2 = R0 * R0 + R1 * R1 + R2 * R2
                    r = math.sqrt(r2)
                    Rn = R0 * nj[0] + R1 * nj[1] + R2 * nj[2]
                    kr = k * r
                    E = complex(math.cos(kr), math.sin(kr))
                    w = _W3[q] * area[j] / FOURPI
                    f = E * complex(-1.0, kr) / (r2 * r)
                    accp += w * f * Rn
                    if want_grad:
                        h = -E * complex(3.0 - kr * kr, -3.0 * kr) / (r2 * r2 * r) * Rn
                        g0 += w * (h * R0 - f * nj[0])
                        g1 += w * (h * R1 - f * nj[1])
                        g2 += w * (h * R2 - f * nj[2])
            Dp[i, j] = accp
            if want_grad:
                Dg[i, 0, j] = g0; Dg[i, 1, j] = g1; Dg[i, 2, j] = g2
    return Dp, Dg


@njit(parallel=True, cache=True)
def winding_number(pts, tri):
    """Generalised winding number (1 inside a closed outward-oriented mesh, 0 outside)."""
    n = pts.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for j in range(tri.shape[0]):
            s += _solid_angle(pts[i], tri[j, 0], tri[j, 1], tri[j, 2])
        out[i] = s / FOURPI
    return out
