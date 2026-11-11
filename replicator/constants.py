"""Physical constants and material properties (SI units throughout).

Every number here is an *input assumption* of the feasibility study. Values that
came from literature are tagged with their source in docs/ASSUMPTIONS.md.
"""
import numpy as np

# --- air (20 C, 1 atm) -------------------------------------------------------
C0 = 343.0            # speed of sound [m/s]
RHO0 = 1.204          # density [kg/m^3]
MU_AIR = 1.81e-5      # dynamic viscosity [Pa s]
NU_AIR = MU_AIR / RHO0

# --- drive -------------------------------------------------------------------
F0 = 40_000.0         # transducer frequency [Hz]
OMEGA = 2 * np.pi * F0
K = OMEGA / C0        # wavenumber [rad/m]
LAMBDA = C0 / F0      # wavelength [m]  (8.575 mm)

G_ACC = 9.81          # gravity [m/s^2]
GVEC = np.array([0.0, 0.0, -G_ACC])

# --- transducer (Murata MA40S4S class, 10 mm can) ------------------------------
PISTON_RADIUS = 4.5e-3     # effective radiating radius [m] (Marzo 2015)
P_REF_PER_VPP = 0.17       # far-field on-axis amplitude [Pa*m per V peak-to-peak]
PITCH = 10.5e-3            # centre-to-centre spacing on the PCB [m]

# --- resin (low-viscosity UV-curable acrylate, jettable grade) ----------------
RHO_RESIN = 1100.0         # [kg/m^3]
SIGMA_RESIN = 0.030        # surface tension [N/m]  (UV inkjet inks: 28-33 mN/m)
MU_RESIN = 0.030           # viscosity [Pa s]

# --- droplet integrity ----------------------------------------------------------
# Acoustic Bond number, literature convention  Ba = 2 rho0 v_rms^2 R / sigma.
# Atomisation observed at 2.33 +- 0.59, stable at 1.85 +- 0.47 (Foresti 2013);
# theory 2.5-3.6 (Lee, Anilkumar & Wang).  Design limit:
BA_DESIGN = 1.0
C_RESIN = 1400.0           # speed of sound in liquid resin [m/s]


def gorkov_coeffs(radius, rho_p=RHO_RESIN, c_p=C_RESIN):
    """Return (c1, c2, volume) such that  U = c1*|p|^2 - c2*|grad p|^2.

    Gor'kov potential for a small compressible sphere, complex amplitudes,
    exp(-i w t) convention:
        U = V [ f1 |p|^2 / (4 rho0 c0^2)  -  3 f2 rho0 |v|^2 / 8 ],
        v = grad p / (i w rho0)
    """
    vol = 4.0 / 3.0 * np.pi * radius ** 3
    f1 = 1.0 - (RHO0 * C0 ** 2) / (rho_p * c_p ** 2)
    f2 = 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0)
    c1 = vol * f1 / (4.0 * RHO0 * C0 ** 2)
    c2 = vol * 3.0 * f2 / (8.0 * RHO0 * OMEGA ** 2)
    return c1, c2, vol
