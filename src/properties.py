"""
Thermophysical-property models for the hybrid-nanofluid evacuated-tube-collector study.

Component 1 is the FIRST-NAMED species of each pair (e.g. Al2O3 in 'Al2O3-Cu'); the mass
share s1 is the mass fraction of component 1. Particle properties are constant over the
range considered; carrier properties are temperature-dependent (T in degrees C).

Values are representative literature data (see README and the manuscript property tables).
"""
import numpy as np

# Solid nanoparticles: rho [kg m^-3], cp [J kg^-1 K^-1], k [W m^-1 K^-1]
PARTICLES = {
    "Al2O3":    dict(rho=3970.0, cp=765.0, k=40.0),
    "Cu":       dict(rho=8933.0, cp=385.0, k=401.0),
    "MWCNT":    dict(rho=2100.0, cp=711.0, k=3000.0),
    "Fe3O4":    dict(rho=5180.0, cp=670.0, k=9.7),
    "Graphene": dict(rho=2250.0, cp=717.0, k=5000.0),
    "TiO2":     dict(rho=4250.0, cp=686.0, k=8.9),
}
# Sphericity psi (Hamilton-Crosser shape factor n = 3/psi); spheres = 1.0
SPHERICITY = {"Al2O3":1.0, "Cu":1.0, "TiO2":1.0, "Fe3O4":1.0, "MWCNT":0.30, "Graphene":0.20}

# Hybrid pairs: component 1 = first-named species, component 2 = second-named species
PAIRS = {
    "Al2O3-Cu":     ("Al2O3", "Cu"),
    "MWCNT-Fe3O4":  ("MWCNT", "Fe3O4"),
    "Graphene-TiO2":("Graphene", "TiO2"),
}

def water_props(T):
    """Distilled water (T in C): returns (rho, cp, k, mu) in SI."""
    rho = 1000.6 - 0.0128*T - 0.0035*T**2
    cp  = 4180.0 + 0.0*T
    k   = 0.5582 + 0.00214*T - 9.5e-6*T**2
    mu  = 2.414e-5 * 10.0**(247.8/((T + 273.15) - 140.0))   # Vogel form: A*10^(B/(T+C)), C=133.15 in degC
    return rho, cp, k, mu

def egw_props(T):
    """60:40 ethylene-glycol/water by mass (T in C)."""
    rho = 1093.0 - 0.55*T
    cp  = 3280.0 + 3.0*T
    k   = 0.380 + 0.0005*T
    mu  = 6.93e-3 * np.exp(-0.0216*T)
    return rho, cp, k, mu

def oil_props(T):
    """Synthetic heat-transfer oil (T in C)."""
    rho = 900.0 - 0.70*T
    cp  = 1800.0 + 3.5*T
    k   = 0.137 - 8e-5*T
    mu  = 5.77e-2 * np.exp(-0.0339*T)
    return rho, cp, k, mu

CARRIERS = {
    "Distilled water":   water_props,
    "EG/water 60:40":    egw_props,
    "Synthetic HTF oil": oil_props,
}

# Fixed geometry / optics (single representative absorber tube; module aperture A_c)
GEOM = dict(di=0.008, L=1.8, Ac=2.0, tau_alpha=0.80, UL=1.8, eta_pump=0.70, d_particle=40e-9)
GEOM["As"] = np.pi * GEOM["di"] * GEOM["L"]   # 0.0452 m^2

# Full-factorial design space (controlled inputs)
DESIGN = dict(
    w_pct      = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],   # total mass fraction (%)
    s1_pct     = [25, 50, 75],                            # component-1 mass share (%)
    Vdot_lpm   = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],# per-tube volumetric flow (L/min)
    Ti_C       = [20, 40, 60, 80],
    Is_Wm2     = [200, 700, 1200],
    Ta_C       = [15, 25, 35],
)
# Optimisation bounds and constraint caps
BOUNDS = dict(w=(0.25, 3.0), s1=(0.25, 0.75), Vdot=(0.5, 6.0))   # s1 fractional
PHI_MAX = 0.012
DP_MAX_KPA = 18.0
REP_POINT = dict(Ti=60.0, Is=700.0, Ta=25.0)                     # representative operating point
