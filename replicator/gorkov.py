"""Acoustic radiation force on a small droplet (Gor'kov) and trap metrics."""
import numpy as np
from . import constants as C


class Droplet:
    def __init__(self, diameter, rho=C.RHO_RESIN, sigma=C.SIGMA_RESIN, mu=C.MU_RESIN):
        self.d = diameter
        self.a = diameter / 2
        self.rho = rho
        self.sigma = sigma
        self.mu = mu
        self.c1, self.c2, self.vol = C.gorkov_coeffs(self.a, rho)
        self.mass = rho * self.vol
        self.weight = self.mass * C.G_ACC

    def potential(self, p, g):
        """Gor'kov potential from complex pressure p (...,) and gradient g (...,3)."""
        return self.c1 * np.abs(p) ** 2 - self.c2 * np.sum(np.abs(g) ** 2, axis=-1)

    def bond_number(self, p, g):
        """Acoustic Bond number, literature convention Ba = 2 rho0 v_rms^2 R / sigma.

        To also guard against a droplet sitting in a pressure antinode, the
        energy density used is the larger of rho0 <v^2> and <p^2>/(rho0 c0^2).
        """
        ep = np.abs(p) ** 2 / (2 * C.RHO0 * C.C0 ** 2)
        ek = np.sum(np.abs(g) ** 2, axis=-1) / (2 * C.RHO0 * C.OMEGA ** 2)
        return 2.0 * np.maximum(ep, ek) * self.a / self.sigma


# ---- finite-difference stencils ---------------------------------------------
def stencil7(h):
    o = np.zeros((7, 3))
    for d in range(3):
        o[1 + 2 * d, d] = h
        o[2 + 2 * d, d] = -h
    return o


def stencil19(h):
    """centre, +-axes (6), and the 12 in-plane diagonals for cross derivatives."""
    o = [np.zeros(3)]
    for d in range(3):
        for s in (1, -1):
            e = np.zeros(3); e[d] = s * h; o.append(e)
    for (a, b) in ((0, 1), (0, 2), (1, 2)):
        for sa in (1, -1):
            for sb in (1, -1):
                e = np.zeros(3); e[a] = sa * h; e[b] = sb * h; o.append(e)
    return np.array(o)


def _fd_operators(h):
    """Linear maps from the 19 stencil potentials to force (3,19) and Hessian (6,19).
    Hessian order: xx, yy, zz, xy, xz, yz."""
    Fop = np.zeros((3, 19))
    Hop = np.zeros((6, 19))
    for d in range(3):
        Fop[d, 1 + 2 * d] = -1 / (2 * h)
        Fop[d, 2 + 2 * d] = +1 / (2 * h)
        Hop[d, 1 + 2 * d] = 1 / h ** 2
        Hop[d, 2 + 2 * d] = 1 / h ** 2
        Hop[d, 0] = -2 / h ** 2
    base = 7
    for n, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        i = base + 4 * n
        # order: (+,+), (+,-), (-,+), (-,-)
        Hop[3 + n, i + 0] = 1 / (4 * h ** 2)
        Hop[3 + n, i + 1] = -1 / (4 * h ** 2)
        Hop[3 + n, i + 2] = -1 / (4 * h ** 2)
        Hop[3 + n, i + 3] = 1 / (4 * h ** 2)
    return Fop, Hop


def hess_matrix(h6):
    return np.array([[h6[0], h6[3], h6[4]], [h6[3], h6[1], h6[5]], [h6[4], h6[5], h6[2]]])


class TrapStencil:
    """Field rows on a 19-point stencil around x: everything the hologram solver
    and the trap diagnostics need, as quadratic forms in the drives w."""

    def __init__(self, field, x, drop: Droplet, h=0.1e-3):
        self.x = np.asarray(x, float)
        self.h = h
        self.drop = drop
        self.A, self.G = field.rows(self.x + stencil19(h))       # (19,T), (19,3,T)
        self.Fop, self.Hop = _fd_operators(h)

    def potentials(self, w):
        P = self.A @ w
        Gv = self.G @ w
        return self.drop.c1 * np.abs(P) ** 2 - self.drop.c2 * np.sum(np.abs(Gv) ** 2, axis=1), P, Gv

    def metrics(self, w):
        """Force [N], Hessian of U [N/m] (3x3), centre pressure, Bond number."""
        U, P, Gv = self.potentials(w)
        F = self.Fop @ U
        H = hess_matrix(self.Hop @ U)
        return dict(F=F, H=H, p=P[0], grad=Gv[0], U=U[0],
                    Ba=float(self.drop.bond_number(P[0], Gv[0])))

    def dU_dphi(self, w, P, Gv, coeff):
        """Gradient wrt transducer phases of  sum_s coeff[s] * U_s."""
        d = self.drop
        # q[s,t] = c1 conj(P_s) A_st - c2 sum_d conj(G_sd) G_sdt ;  dU_s/dphi_t = -2 Im(w_t q_st)
        q = d.c1 * (np.conj(P) * coeff)[:, None] * self.A
        q -= d.c2 * np.einsum("sd,sdt->st", np.conj(Gv) * coeff[:, None], self.G)
        return -2.0 * np.imag(w * q.sum(0))
