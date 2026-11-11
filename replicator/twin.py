"""Digital-twin calibration: turn measurements of the real machine into corrections
of the simulation's model, so the hologram planner works with what is actually there.

Three measurements, all from sensors that are on the machine:

1. Array self-calibration (MEMS microphone on every board).  Fire each transducer
   alone; every other board's microphone records amplitude and phase.  Comparing
   with the model's prediction gives a complex gain per transducer (its real
   output and phase lag) and the speed of sound.
2. Trap test (cameras).  Hold a droplet still; its measured resting position minus
   the model's prediction, times the trap stiffness, is the force the model does
   not know about -- e.g. acoustic streaming.  Ring-down frequency checks stiffness.
3. Flight log (cameras).  Every droplet's measured 3-D track is stored next to the
   planned one in the same format the viewer replays.
"""
import numpy as np
from . import constants as C
from .dynamics import Keys, Plant, simulate


def mic_positions(array, u_mm=20.5, v_mm=-5.25):
    """One microphone per board: on the PCB at board coordinates (u, v), its port in the board face."""
    return np.array([p.center + u_mm * 1e-3 * p.u + v_mm * 1e-3 * p.v for p in array.panels])


def measure_array(true_array, mics, noise=0.01, rng=None):
    """What the microphones record: complex pressure at each mic for each transducer fired alone."""
    rng = rng or np.random.default_rng(0)
    Y, _ = true_array.rows(mics, grad=False)
    scale = np.abs(Y).mean()
    return Y + noise * scale * (rng.normal(size=Y.shape) + 1j * rng.normal(size=Y.shape)) / np.sqrt(2)


def calibrate_array(model_array, mics, Y, k_span=0.01, n_k=41):
    """Estimate the sound speed (via k) and a complex gain per transducer.

    For a trial k the model predicts A[m,t]; the least-squares gain of transducer t
    is g_t = <A_t, Y_t> / <A_t, A_t> over the microphones of *other* boards.  The
    k that leaves the smallest residual is kept."""
    import copy
    arr = copy.copy(model_array)
    arr.gain = np.ones(arr.n)
    k0 = model_array.k
    own = np.arange(len(mics))[:, None] == arr.panel_id[None, :]
    best = None
    for k in k0 * (1 + np.linspace(-k_span, k_span, n_k)):
        arr.k = k
        A, _ = arr.rows(mics, grad=False)
        A = np.where(own, 0, A)
        Ym = np.where(own, 0, Y)
        g = np.sum(np.conj(A) * Ym, axis=0) / np.sum(np.abs(A) ** 2, axis=0)
        res = np.linalg.norm(Ym - A * g) / np.linalg.norm(Ym)
        if best is None or res < best[0]:
            best = (res, k, g)
    res, k, g = best
    g = g / np.exp(1j * np.angle(np.sum(g)))          # the common phase is arbitrary
    return dict(k=k, gain=g, residual=res, c0=C.OMEGA / k)


def apply_calibration(array, cal):
    """Model array corrected with the measured gains and sound speed."""
    import copy
    out = copy.copy(array)
    out.k = cal["k"]
    out.gain = cal["gain"]
    return out


def trap_test(plant: Plant, planner, x, duration=0.3, cam_hz=1000.0, cam_noise=15e-6, rng=None):
    """Hold a droplet at x and watch it with the cameras.  Returns the unexplained
    steady force (N) and the wind that would produce it."""
    rng = rng or np.random.default_rng(0)
    drop = planner.drop
    k = planner.hold(x, duration, None, 0.5)
    info = k.info[0]
    r = simulate(plant, drop, x + info["offset"], np.zeros(3), k, t_extra=0.0, record_every=max(1, int(1e4 / cam_hz)))
    tr = r["traj"]
    meas = tr[:, 1:4] + cam_noise * rng.normal(size=(len(tr), 3))
    half = len(meas) // 2
    dx = meas[half:].mean(0) - (x + info["offset"])
    F = info["H"] @ dx                                 # restoring force is -H dx, so the push is +H dx
    c = 3 * np.pi * C.MU_AIR * drop.d
    sig = meas[:, 0] - meas[:, 0].mean()
    f = np.fft.rfftfreq(len(sig), tr[1, 0] - tr[0, 0])
    return dict(offset=dx, force=F, wind=F / c, kappa_model=info["kappa"], freq_model=info["freq_hz"],
                freq_meas=float(f[np.abs(np.fft.rfft(sig))[1:].argmax() + 1]))
