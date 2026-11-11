"""Study 12: closing the loop between the built machine and the simulation.

A 'real' machine is simulated with things the model does not know: generic
transducers with 14 deg phase scatter and 15 % amplitude scatter, air 1.5 K warmer
than assumed, and a steady 0.2 m/s streaming current.  We then do what a
technician would do with the on-board sensors and see whether landing improves:
  step 0  plan with the nominal model
  step 1  array self-calibration with the board microphones -> gains + sound speed
  step 2  camera trap test -> unexplained force -> streaming estimate -> feed-forward
"""
import sys, time, json
import numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from replicator import constants as C, configs, status
from replicator.acoustics import Mesh, Scatterer, Field
from replicator.gorkov import Droplet
from replicator.scene import new_part, build_field
from replicator.geometry import refine_near
from replicator.scene import DEFAULT_LEVELS
from replicator.dynamics import Planner, Plant, simulate
from replicator.delivery import Site, Jet, JET_MK2, plan_delivery
from replicator import twin

rng = np.random.default_rng(7)
drop = Droplet(0.3e-3)
R = 10e-3
part = new_part(extent=16e-3, pole_tip_z=R - 2e-3, res=0.2e-3)
part.add_sphere([0, 0, 0], R)
n = np.array([1.0, 0, 0])
xs = part.project((n * R)[None])[0]
site = Site(xs, part.normal(xs[None])[0])
jet = Jet(**JET_MK2)
mesh = refine_near(part.to_mesh(), xs, DEFAULT_LEVELS, project=part.project)

# ---- the 'real' machine --------------------------------------------------------
true = configs.REGISTRY["rco5"]()
true.k = C.K * (1 - 1.5 / (2 * 293.15))
true.gain = (1 + 0.15 * rng.normal(size=true.n)) * np.exp(1j * np.radians(14) * rng.normal(size=true.n))
wind_true = np.array([0.12, -0.14, 0.08])
plant = Plant(Field(true, Scatterer(mesh, true)), phase_levels=128, wind=wind_true)

nominal = configs.REGISTRY["rco5"]()
mics = twin.mic_positions(nominal)


def fly(model_array, label, wind_ff=None):
    fld = Field(model_array, Scatterer(mesh, model_array))
    pl = Planner(fld, drop, ds=0.15e-3)
    if wind_ff is not None:
        pl.wind = wind_ff
    keys, summ = plan_delivery(pl, jet, site, part, np.zeros(3), R)
    errs, lost = [], 0
    r2 = np.random.default_rng(11)
    for _ in range(4):
        x0, v0 = jet.sample(drop, r2, gate=True)
        r = simulate(plant, drop, x0, v0, keys, part=part)
        if r["status"] == "landed":
            errs.append(float(np.linalg.norm(part.project(r["x"][None])[0] - xs)) * 1e6)
        else:
            lost += 1
    print(f"  {label:44s}: landed {len(errs)}/4" + (f", error median {np.median(errs):.0f} um (max {max(errs):.0f})" if errs else ""), flush=True)
    return dict(label=label, landed=len(errs), errs=errs), pl


out = {}
print("step 0: plan with the nominal model")
out["nominal"], _ = fly(nominal, "nominal model")

print("step 1: array self-calibration with the board microphones")
Y = twin.measure_array(true, mics, noise=0.02, rng=rng)
cal = twin.calibrate_array(nominal, mics, Y)
g_true = true.gain / np.exp(1j * np.angle(np.sum(true.gain)))
ph_before = np.degrees(np.sqrt(np.mean(np.angle(g_true) ** 2)))
ph_after = np.degrees(np.sqrt(np.mean(np.angle(cal["gain"] / g_true) ** 2)))
am_after = float(np.sqrt(np.mean((np.abs(cal["gain"]) / np.abs(g_true) - 1) ** 2)))
print(f"  sound speed: assumed {C.C0:.2f}, true {C.OMEGA/true.k:.2f}, measured {cal['c0']:.2f} m/s")
print(f"  transducer phase error: {ph_before:.1f} deg rms before -> {ph_after:.1f} deg rms after; amplitude error after {100*am_after:.1f} %")
model1 = twin.apply_calibration(nominal, cal)
out["calibrated"], pl1 = fly(model1, "after microphone calibration")
out["cal"] = dict(c_true=float(C.OMEGA / true.k), c_meas=float(cal["c0"]), phase_before_deg=float(ph_before),
                  phase_after_deg=float(ph_after), amp_after=am_after)

print("step 2: camera trap test for the force the model does not explain")
free_plant = Plant(Field(true), phase_levels=128, wind=wind_true)
plf = Planner(Field(model1), drop)
tt = twin.trap_test(free_plant, plf, np.array([13e-3, 0, -6e-3]), rng=rng)
print(f"  resting offset {np.round(tt['offset']*1e6)} um -> unexplained force {np.round(tt['force']/drop.weight,3)} weights")
print(f"  inferred steady air current {np.round(tt['wind'],3)} m/s (true {wind_true}); ring-down {tt['freq_meas']:.1f} Hz vs model {tt['freq_model'][0]:.1f}-{tt['freq_model'][2]:.1f} Hz")
out["trap_test"] = dict(wind_est=tt["wind"].tolist(), wind_true=wind_true.tolist(), freq_meas=tt["freq_meas"])
out["tuned"], _ = fly(model1, "calibrated + streaming feed-forward", wind_ff=tt["wind"])
json.dump(out, open(f"{ROOT}/out/twin_demo.json", "w"), indent=1, default=float)
med = lambda k: np.median(out[k]["errs"]) if out[k]["errs"] else float("nan")
status.update(test=("twin", "Digital twin: measure the built machine, correct the model, re-plan",
                    "pass" if med("tuned") < med("nominal") else "warn",
                    f"uncalibrated generic transducers: landed {out['nominal']['landed']}/4, median {med('nominal'):.0f} µm\n"
                    f"after board-microphone calibration: {out['calibrated']['landed']}/4, {med('calibrated'):.0f} µm "
                    f"(phase error {ph_before:.0f}° → {ph_after:.1f}°)\n"
                    f"plus camera trap test + streaming feed-forward: {out['tuned']['landed']}/4, {med('tuned'):.0f} µm"))
