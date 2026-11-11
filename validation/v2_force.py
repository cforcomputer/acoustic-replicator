"""Validation of the radiation-force layer.

V4  Gor'kov force in a plane standing wave  vs  F = 4 pi a^3 k E_ac Phi sin(2kz)
V5  minimum levitation pressure for the resin  vs  closed form
V6  hologram-solver analytic gradient  vs  finite differences
V7  free-space sanity: classic 2x(16x16) opposed levitator at 20 Vpp
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from replicator import constants as C
from replicator.gorkov import Droplet, TrapStencil
from replicator.holo import TrapSolver, restoring_profile, holding_force
from replicator.acoustics import Field
from replicator import configs

ok_all = True


def check(name, val, tol):
    global ok_all
    ok = val < tol
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e})")


class PlaneWaves:
    """Mock 'array' of two counter-propagating unit plane waves along z."""
    class _A:
        n = 2
    array = _A()

    def rows(self, pts):
        pts = np.atleast_2d(pts)
        z = pts[:, 2]
        A = np.stack([np.exp(1j * C.K * z), np.exp(-1j * C.K * z)], 1)
        G = np.zeros((len(pts), 3, 2), complex)
        G[:, 2, 0] = 1j * C.K * A[:, 0]
        G[:, 2, 1] = -1j * C.K * A[:, 1]
        return A, G


print("V4 Gor'kov force in a plane standing wave")
drop = Droplet(0.4e-3)
pa = 3000.0                       # standing-wave pressure amplitude
w = np.array([pa / 2, pa / 2], complex)      # p = pa cos(kz)
fld = PlaneWaves()
f1 = 1 - C.RHO0 * C.C0 ** 2 / (drop.rho * C.C_RESIN ** 2)
f2 = 2 * (drop.rho - C.RHO0) / (2 * drop.rho + C.RHO0)
Eac = pa ** 2 / (4 * C.RHO0 * C.C0 ** 2)
Phi = f1 / 3 + f2 / 2
worst = 0
for z in np.linspace(0.1e-3, 4e-3, 9):
    st = TrapStencil(fld, [0, 0, z], drop, h=0.05e-3)
    F = st.metrics(w)["F"][2]
    Fa = 4 * np.pi * drop.a ** 3 * C.K * Eac * Phi * np.sin(2 * C.K * z)
    worst = max(worst, abs(F - Fa) / (4 * np.pi * drop.a ** 3 * C.K * Eac * Phi))
check("max |F_num - F_analytic| / F_max", worst, 2e-3)

print("V5 minimum standing-wave amplitude to levitate the resin")
p_min = np.sqrt(drop.rho * C.G_ACC * 4 * C.RHO0 * C.C0 ** 2 / (3 * C.K * Phi))
st = TrapStencil(fld, [0, 0, C.LAMBDA / 8], drop, h=0.05e-3)
Fmax = st.metrics(np.array([p_min / 2, p_min / 2], complex))["F"][2]
print(f"     p_min = {p_min:.0f} Pa  ({20*np.log10(p_min/np.sqrt(2)/20e-6):.1f} dB SPL)")
check("|F_max(p_min)/weight - 1|", abs(Fmax / drop.weight - 1), 2e-3)

print("V6 hologram objective gradient vs finite differences")
arr = configs.opposed(8, 0.12)
field = Field(arr)
solver = TrapSolver(field, drop, ba_max=0.1)
x = np.array([3e-3, -2e-3, 4e-3])
st = TrapStencil(field, x, drop, solver.h)
rng = np.random.default_rng(0)
phi = np.append(rng.uniform(0, 2 * np.pi, arr.n), 0.7)
from replicator.holo import Term
terms = [Term(st, solver.Pm, solver.f_target)]            # the objective as the planner calls it: one trap term + phase-smoothness term
ref = rng.uniform(0, 2 * np.pi, arr.n)
J, g = solver._objective(phi, terms, ref)
worst = 0
for t in list(rng.choice(arr.n, 12, replace=False)) + [arr.n]:
    e = np.zeros(arr.n + 1); e[t] = 1e-6
    fd = (solver._objective(phi + e, terms, ref)[0] - solver._objective(phi - e, terms, ref)[0]) / 2e-6
    worst = max(worst, abs(fd - g[t]) / (abs(g).max()))
check("max rel gradient error", worst, 1e-5)

print("V7 free-space sanity: 2x(16x16) opposed boards, 200 mm apart, 20 Vpp")
arr = configs.opposed(16, 0.20, vpp=20.0)
field = Field(arr)
x = np.zeros(3)
foc = arr.focus_phases(x)
w = np.exp(1j * foc)
p, _ = field.eval(x[None], w)
print(f"     all-in-phase focus pressure at centre: {abs(p[0]):.0f} Pa "
      f"({20*np.log10(abs(p[0])/np.sqrt(2)/20e-6):.1f} dB SPL) from {arr.n} transducers")
for ba in (0.5, 1.0, 2.0):
    solver = TrapSolver(field, drop, ba_max=ba)
    phi, amp, info = solver.solve_best(x)
    w = amp * np.exp(1j * phi)
    s, f = restoring_profile(field, drop, x, w, np.eye(3))
    hold = [min(holding_force(s, f[i])) for i in range(3)]
    print(f"     Ba<={ba}: init '{info['init']}', amp {amp:.2f}, kappa [W/mm] = {np.round(info['kappa'],2)}, "
          f"freq {np.round(info['freq_hz'],1)} Hz, Ba {info['Ba']:.2f}, |f_res| {np.linalg.norm(info['f_res']):.3f}, "
          f"holding force xyz [W] = {np.round(hold,2)}")
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
