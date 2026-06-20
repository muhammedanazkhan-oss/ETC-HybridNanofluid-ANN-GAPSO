"""
Reduced-order, self-consistent thermo-hydraulic solver for a forced-convection direct-flow
evacuated-tube collector with hybrid-nanofluid working fluids.

Effective properties: Eq.(1)-(12)  (volume fractions, mixing rules, Takabi-Salehi conductivity,
Brinkman viscosity, zero-loading limit).
Flow & convection: Eq.(13)-(23)  (constant-wall-temperature laminar limit Nu=3.66, Hausen
developing-flow, Gnielinski turbulent, linear transition blend over 2300<=Re<3000, h=Nu k/di).
Collector: Eq.(24)-(32)  (Hottel-Whillier-Bliss; effective lumped F').
Hydraulics & PEC: Eq.(33)-(40).
  PEC_f   = (Nu/Nu_bf) / (f/f_bf)^(1/3)                    (classical friction-factor diagnostic)
  PEC_eps = (Nu/Nu_bf) / (Wp_hnf/Wp_bf)^(1/3)              (primary pumping-power-ratio metric)
          = PEC_f / (rho_hnf/rho_bf)^(1/3)                  at equal volumetric flow rate
All arrays are vectorised; scalars also accepted.
"""
import numpy as np
from properties import PARTICLES, PAIRS, CARRIERS, GEOM

di, L, Ac, ta, UL, ep = GEOM["di"], GEOM["L"], GEOM["Ac"], GEOM["tau_alpha"], GEOM["UL"], GEOM["eta_pump"]
As = GEOM["As"]

def nusselt(Re, Pr, lam_base=3.66):
    """Nu by flow regime with linear transition blend. lam_base=3.66 (const-T) or 4.36 (const-q)."""
    x = (di/L)*Re*Pr
    Nu_lam = lam_base + (0.0668*x)/(1.0 + 0.04*x**(2.0/3.0))
    ft = (0.790*np.log(np.maximum(Re, 1e-9)) - 1.64)**-2
    Nu_turb = np.maximum(((ft/8.0)*(Re-1000.0)*Pr)/(1.0 + 12.7*np.sqrt(ft/8.0)*(Pr**(2.0/3.0)-1.0)), lam_base)
    lam = np.clip((Re-2300.0)/700.0, 0.0, 1.0)
    return np.where(Re < 2300.0, Nu_lam, np.where(Re >= 3000.0, Nu_turb, (1.0-lam)*Nu_lam + lam*Nu_turb))

def friction(Re):
    """Darcy friction factor: laminar 64/Re, Petukhov turbulent, linear transition blend."""
    f_lam = 64.0/Re
    f_turb = (0.790*np.log(np.maximum(Re, 1e-9)) - 1.64)**-2
    lam = np.clip((Re-2300.0)/700.0, 0.0, 1.0)
    return np.where(Re < 2300.0, f_lam, np.where(Re >= 3000.0, f_turb, (1.0-lam)*f_lam + lam*f_turb))

def effective_props(pair, carrier, w, s1, Tm, mu_factor=1.0):
    """Eq.(1)-(11) with zero-loading limit Eq.(12). w = total mass fraction (0-1), s1 = mass share of comp 1 (0-1)."""
    fn = CARRIERS[carrier]
    rb, cb, kb, mb = fn(Tm)
    p1 = PARTICLES[PAIRS[pair][0]]; p2 = PARTICLES[PAIRS[pair][1]]
    w1 = s1*w; w2 = (1.0-s1)*w
    denom = (w1/p1["rho"]) + (w2/p2["rho"]) + (1.0-w1-w2)/rb
    phi1 = (w1/p1["rho"])/denom; phi2 = (w2/p2["rho"])/denom; phi = phi1 + phi2
    rho = np.where(phi > 1e-12, (1.0-phi)*rb + phi1*p1["rho"] + phi2*p2["rho"], rb)
    rhocp = (1.0-phi)*(rb*cb) + phi1*(p1["rho"]*p1["cp"]) + phi2*(p2["rho"]*p2["cp"])
    cp = np.where(phi > 1e-12, rhocp/rho, cb)
    P = phi1*p1["k"] + phi2*p2["k"]                       # phi_p k_p
    safe = np.maximum(phi, 1e-12)
    num = (P/safe) + 2.0*kb + 2.0*P - 2.0*phi*kb
    den = (P/safe) + 2.0*kb - P + phi*kb
    k = np.where(phi > 1e-12, kb*num/den, kb)             # Takabi-Salehi
    mu = np.where(phi > 1e-12, mu_factor*mb*(1.0-phi)**(-2.5), mb)   # Brinkman
    return dict(rho=rho, cp=cp, k=k, mu=mu, phi=phi, rho_bf=rb, cp_bf=cb, k_bf=kb, mu_bf=mb)

def solve(pair, carrier, w, s1, Vdot_lpm, Ti, Is, Ta,
          niter=25, lam_base=3.66, mu_factor=1.0, sumK=0.0):
    """Self-consistent solve. w,s1 fractional. Returns dict of arrays (PEC_f, PEC_eps, Nu, f, dP[kPa], Wp, eta, phi, Re, Pr, ...)."""
    w = np.asarray(w, float); s1 = np.asarray(s1, float)
    Vd = np.asarray(Vdot_lpm, float)/60000.0                  # L/min -> m^3/s
    Ti = np.asarray(Ti, float); Is = np.asarray(Is, float); Ta = np.asarray(Ta, float)
    Tm = np.broadcast_arrays(w, s1, Vd, Ti, Is, Ta)[3].astype(float).copy()
    for _ in range(niter):
        pr = effective_props(pair, carrier, w, s1, Tm, mu_factor)
        rho, cp, k, mu = pr["rho"], pr["cp"], pr["k"], pr["mu"]
        rb, cb, kb, mb = pr["rho_bf"], pr["cp_bf"], pr["k_bf"], pr["mu_bf"]
        u = 4.0*Vd/(np.pi*di**2)
        Re = rho*u*di/mu; Pr = mu*cp/k
        Re_bf = rb*u*di/mb; Pr_bf = mb*cb/kb
        Nu = nusselt(Re, Pr, lam_base); f = friction(Re)
        Nu_bf = nusselt(Re_bf, Pr_bf, lam_base); f_bf = friction(Re_bf)
        h = Nu*k/di; mdot = rho*Vd
        Fp = h*As/(h*As + UL*Ac)
        FR = (mdot*cp)/(Ac*UL)*(1.0 - np.exp(-Ac*UL*Fp/(mdot*cp)))
        Qu = np.maximum(Ac*FR*(ta*Is - UL*(Ti-Ta)), 0.0)
        To = Ti + Qu/(mdot*cp)
        Tm_new = (Ti + To)/2.0
        if np.max(np.abs(Tm_new - Tm)) < 1e-4:
            Tm = Tm_new; break
        Tm = Tm_new
    dP = (f*(L/di) + sumK)*(rho*u**2/2.0)
    dP_bf = (f_bf*(L/di) + sumK)*(rb*u**2/2.0)
    Wp = Vd*dP/ep; Wp_bf = Vd*dP_bf/ep
    eta = Qu/(Ac*Is)
    PEC_f = (Nu/Nu_bf)/(f/f_bf)**(1.0/3.0)
    PEC_eps = (Nu/Nu_bf)/(Wp/Wp_bf)**(1.0/3.0)
    return dict(PEC_f=PEC_f, PEC_eps=PEC_eps, Nu=Nu, Nu_bf=Nu_bf, f=f, f_bf=f_bf,
                dP_kPa=dP/1000.0, Wp=Wp, eta=eta, phi=pr["phi"], Re=Re, Pr=Pr,
                rho=rho, rho_bf=rb, h=h, To=To, Tm=Tm, u=u)

def penalty(phi, dP_kPa, lam_phi=10.0, lam_P=10.0, phi_max=0.012, dP_max=18.0):
    """Static quadratic penalty (Algorithm 1 / Eq. for F)."""
    pp = np.maximum(0.0, (phi - phi_max)/phi_max)**2
    pP = np.maximum(0.0, (dP_kPa - dP_max)/dP_max)**2
    return lam_phi*pp + lam_P*pP
