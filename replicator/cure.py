"""UV cure budget for the machine as drawn: how much 405 nm light reaches a landing
site from the eleven emitters in the side ports, and how long the flash must be.

Working-curve model: cure depth  Cd = Dp ln(E / Ec).  A droplet's shape is frozen
once the dose at its contact line reaches Ec (gel point); curing it through takes
the dose for the full blob thickness, which also accumulates from later flashes
because every flash lights the whole build zone.
"""
import numpy as np

# 405 nm working-curve parameters measured by Bennett 2017 (Addit. Manuf.): Dp [um], Ec [mJ/cm^2]
RESINS_405 = {
    "PR48": (53.0, 6.3),
    "Formlabs Clear": (192.0, 12.6),
    "VeroWhitePlus": (145.0, 1.9),
    "TangoBlackPlus": (151.0, 4.1),
    "VeroClear": (568.0, 6.9),
}
# emitter as sourced: Luminus SST-10-UV-A130-G405 (>= 0.9 W at 0.5 A; about 1.2 W at the 0.7 A the LDD-700L driver supplies)
# behind a Ledil LISA2 10 mm lens (15-24 degrees full width at half maximum, depending on the LED).
EMITTER = dict(flux_w=1.2, lens_efficiency=0.85, fwhm_deg=20.0)


def dose_for_depth(depth_um, Dp_um, Ec):
    """Exposure [mJ/cm^2] that gels the resin to `depth_um`:  Cd = Dp ln(E/Ec)."""
    return Ec * np.exp(depth_um / Dp_um)


def irradiance_at(x, n, lights, em=EMITTER):
    """Irradiance [W/cm^2] on a surface element at x (m) with outward normal n, from Gaussian beams."""
    E = 0.0
    hw = np.radians(em["fwhm_deg"] / 2)
    for l in lights:
        p, a = np.array(l["pos"]), np.array(l["dir"])
        r = x - p
        z = float(r @ a)
        if z <= 0:
            continue
        rho = np.linalg.norm(r - z * a)
        w = z * np.tan(hw)                                   # half width at half maximum at this distance
        e0 = em["flux_w"] * em["lens_efficiency"] * np.log(2) / (np.pi * (w * 100) ** 2)     # peak, W/cm^2
        cosi = float(n @ (-r / np.linalg.norm(r)))
        if cosi > 0:
            E += e0 * 2.0 ** (-(rho / w) ** 2) * cosi
    return E


def machine_budget(blob_depth_um=120.0, n_dir=400, seed=0):
    """Irradiance on surface elements of every orientation, at the build centre and 10 mm off it."""
    from .cad import Machine
    lights = [l for l in Machine().build()["lights"] if l["kind"] == "uv"]
    rng = np.random.default_rng(seed)
    nn = rng.normal(size=(n_dir, 3)); nn /= np.linalg.norm(nn, axis=1)[:, None]
    out = {}
    for name, pts in (("centre", [np.zeros(3)]), ("10 mm off centre", [0.010 * v for v in np.vstack([np.eye(3), -np.eye(3)])])):
        E = np.array([irradiance_at(x, n, lights) for x in pts for n in nn])
        out[name] = dict(min=float(E.min()), p10=float(np.percentile(E, 10)), median=float(np.median(E)), max=float(E.max()))
    rows = []
    e_typ, e_low = out["10 mm off centre"]["median"], out["10 mm off centre"]["p10"]
    for nm, (Dp, Ec) in list(RESINS_405.items()) + [("typical hobby resin (assumed Ec 4, Dp 120)", (120.0, 4.0))]:
        full = dose_for_depth(blob_depth_um, Dp, Ec)
        rows.append(dict(resin=nm, Ec=Ec, Dp_um=Dp, gel_ms_typ=Ec / e_typ, gel_ms_low=Ec / e_low, through_dose=float(full), through_ms_typ=float(full / e_typ)))
    return dict(n_emitters=len(lights), emitter=EMITTER, irradiance_w_cm2=out, blob_depth_um=blob_depth_um, rows=rows)


if __name__ == "__main__":
    import json, sys
    r = machine_budget()
    print(f"{r['n_emitters']} emitters, {r['emitter']['flux_w']} W each, {r['emitter']['fwhm_deg']} deg beams")
    for k, v in r["irradiance_w_cm2"].items():
        print(f"  irradiance {k:17s}: median {v['median']:.2f} W/cm2 over all surface orientations (10 % of orientations below {v['p10']:.2f}, minimum {v['min']:.2f}, maximum {v['max']:.2f})")
    for x in r["rows"]:
        print(f"  {x['resin']:44s} Ec {x['Ec']:5.1f} mJ/cm2 -> shape frozen after {x['gel_ms_typ']:5.1f} ms (worst orientations {x['gel_ms_low']:5.1f} ms); "
              f"through-cure of a {r['blob_depth_um']:.0f} um blob needs {x['through_dose']:5.1f} mJ/cm2 = {x['through_ms_typ']:5.0f} ms of light in total")
    if len(sys.argv) > 1:
        json.dump(r, open(sys.argv[1], "w"), indent=1)
