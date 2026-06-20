"""
Model-form sensitivity checks on the global Al2O3-Cu / EG-water optimum (Section 4.5):
  - boundary condition: constant-wall-temperature (Nu_lam=3.66) vs constant-heat-flux (Nu_lam=4.36)
  - viscosity closure : Brinkman hybrid viscosity perturbed by +/-30% (mu_hnf^+- = mu_Brinkman*(1+-0.30))
  - loop minor losses : tube-only vs sum(K_i) = 0,10,20,40
  - matched equal-pumping-power PEC_eq: adjust the base-fluid flow so Wp_bf(Vbf*) = Wp_hnf(Vhnf)
Writes results/sensitivity.json.
"""
import os, json, numpy as np
from properties import CARRIERS, GEOM
from solver import solve, friction, nusselt

HERE=os.path.dirname(__file__); RESULTS=os.path.join(HERE,"..","results")
di,L,ep=GEOM["di"],GEOM["L"],GEOM["eta_pump"]
OPT=dict(pair="Al2O3-Cu", carrier="EG/water 60:40", w=0.03, s1=0.25, V=1.71, Ti=60., Is=700., Ta=25.)

def pec(**kw):
    a=dict(OPT); a.update(kw)
    r=solve(a["pair"],a["carrier"],a["w"],a["s1"],a["V"],a["Ti"],a["Is"],a["Ta"],
            lam_base=a.get("lam_base",3.66), mu_factor=a.get("mu_factor",1.0), sumK=a.get("sumK",0.0))
    return float(r["PEC_eps"]), float(r["PEC_f"]), float(r["Re"])

def matched_pumping():
    r=solve(OPT["pair"],OPT["carrier"],OPT["w"],OPT["s1"],OPT["V"],OPT["Ti"],OPT["Is"],OPT["Ta"])
    Wph=float(r["Wp"]); Nuh=float(r["Nu"]); fn=CARRIERS[OPT["carrier"]]
    def wpb(V):
        Vd=V/60000.; u=4*Vd/(np.pi*di**2); rb,cb,kb,mb=fn(60.); Reb=rb*u*di/mb
        fb=float(friction(np.array([Reb]))[0]); return Vd*fb*(L/di)*(rb*u**2/2.)/ep
    lo,hi=0.05,30.
    for _ in range(80):
        m=0.5*(lo+hi)
        if wpb(m)<Wph: lo=m
        else: hi=m
    Vbf=0.5*(lo+hi); Vd=Vbf/60000.; u=4*Vd/(np.pi*di**2); rb,cb,kb,mb=fn(60.); Reb=rb*u*di/mb; Prb=mb*cb/kb
    Nubf=float(nusselt(np.array([Reb]),np.array([Prb]))[0])
    return dict(Vbf_star=round(Vbf,3), Nu_hnf=round(Nuh,3), Nu_bf_star=round(Nubf,3), PEC_eq=round(Nuh/Nubf,5))

def main():
    os.makedirs(RESULTS,exist_ok=True)
    base_e,base_f,Re=pec()
    out={"optimum_point":OPT, "baseline":dict(PEC_eps=round(base_e,5),PEC_f=round(base_f,5),Re=round(Re)),
         "boundary_condition":{}, "viscosity_closure":{}, "minor_losses":{}, "matched_pumping":matched_pumping()}
    e_q,_,_=pec(lam_base=4.36)
    out["boundary_condition"]={"Nu_lam_3.66":round(base_e,5),"Nu_lam_4.36":round(e_q,5),"dPEC":round(e_q-base_e,5)}
    for mf in [0.7,1.0,1.3]:
        out["viscosity_closure"][f"mu_x{mf}"]=round(pec(mu_factor=mf)[0],5)
    for K in [0,10,20,40]:
        out["minor_losses"][f"sumK_{K}"]=round(pec(sumK=float(K))[0],5)
    json.dump(out,open(os.path.join(RESULTS,"sensitivity.json"),"w"),indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
