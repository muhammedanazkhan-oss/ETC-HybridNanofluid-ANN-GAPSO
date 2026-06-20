"""
Optimisation of the primary pumping-power-ratio criterion PEC_eps for each hybrid pair and carrier.

Fitness  F(x) = PEC_eps(x) - quadratic penalty   (Algorithm 1; phi_max=0.012, dP_max=18 kPa, lambda=10)
Decision vector x = (w[%], s1[fraction], Vdot[L/min]) at the representative point (Ti=60, Is=700, Ta=25).

Produces:
  results/optima_PEC_f.csv    - classical friction-factor diagnostic optima (Table 10)
  results/optima_PEC_eps.csv  - directly re-optimised pumping-power-ratio optima (Table 11)
  results/algorithm_comparison.csv - GA vs PSO vs memetic GA-PSO (generations / evals to converge)
Methods: deterministic dense-grid search, real-coded GA, PSO, and a memetic GA-PSO (PSO refinement
of the GA elites). Default uses the reduced-order solver as the objective; the trained ANN surrogate
can be used instead with --surrogate (PEC_f from the ANN, PEC_eps via the density correction).
"""
import os, json, argparse, numpy as np, pandas as pd
from properties import PAIRS, CARRIERS, BOUNDS, PHI_MAX, DP_MAX_KPA, REP_POINT
from solver import solve, penalty

HERE=os.path.dirname(__file__); RESULTS=os.path.join(HERE,"..","results")
LO=np.array([BOUNDS["w"][0], BOUNDS["s1"][0], BOUNDS["Vdot"][0]])
HI=np.array([BOUNDS["w"][1], BOUNDS["s1"][1], BOUNDS["Vdot"][1]])

def evaluate(pair, carrier, Xm):
    """Xm rows = (w%, s1_frac, Vdot). Returns dict of arrays."""
    Xm=np.atleast_2d(Xm)
    return solve(pair, carrier, Xm[:,0]/100.0, Xm[:,1], Xm[:,2],
                 np.full(len(Xm),REP_POINT["Ti"]), np.full(len(Xm),REP_POINT["Is"]), np.full(len(Xm),REP_POINT["Ta"]))

def fitness(pair, carrier, Xm, metric="PEC_eps"):
    r=evaluate(pair,carrier,Xm)
    return r[metric] - penalty(r["phi"], r["dP_kPa"], phi_max=PHI_MAX, dP_max=DP_MAX_KPA)

def dense_grid(pair, carrier, metric="PEC_eps", nw=28, ns=26, nv=45):
    w=np.linspace(LO[0],HI[0],nw); s=np.linspace(LO[1],HI[1],ns); v=np.linspace(LO[2],HI[2],nv)
    W,S,V=np.meshgrid(w,s,v,indexing="ij"); Xm=np.column_stack([W.ravel(),S.ravel(),V.ravel()])
    r=evaluate(pair,carrier,Xm); val=r[metric]-penalty(r["phi"],r["dP_kPa"],phi_max=PHI_MAX,dP_max=DP_MAX_KPA)
    i=int(np.nanargmax(val))
    return dict(w=Xm[i,0], s1=Xm[i,1], Vdot=Xm[i,2], phi=float(r["phi"][i]), dP_kPa=float(r["dP_kPa"][i]),
                PEC_f=float(r["PEC_f"][i]), PEC_eps=float(r["PEC_eps"][i]), nfev=int(Xm.shape[0]))

def run_pso(pair, carrier, metric="PEC_eps", npart=20, iters=60, seed=0, tol=1e-8, q=5):
    rng=np.random.RandomState(seed); X=LO+rng.rand(npart,3)*(HI-LO); Vv=np.zeros((npart,3)); nf=0
    pv=fitness(pair,carrier,X,metric); nf+=npart; pb=X.copy(); g=pb[pv.argmax()].copy(); gv=pv.max(); hist=[gv]
    gc=None
    for it in range(1,iters+1):
        r1,r2=rng.rand(npart,3),rng.rand(npart,3)
        Vv=0.6*Vv+1.5*r1*(pb-X)+1.5*r2*(g-X); X=np.clip(X+Vv,LO,HI)
        v=fitness(pair,carrier,X,metric); nf+=npart; im=v>pv; pb[im]=X[im]; pv[im]=v[im]
        if pv.max()>gv: gv=pv.max(); g=pb[pv.argmax()].copy()
        hist.append(gv)
        if gc is None and it>q and abs(hist[it]-hist[it-q])<tol: gc=it
    return dict(best=g, val=float(gv), gen_conv=gc if gc else iters, nfev=nf, hist=hist)

def run_ga(pair, carrier, metric="PEC_eps", npop=60, gens=100, seed=0, tol=1e-8, q=5, memetic=False, ne=6, npso=5):
    rng=np.random.RandomState(seed); X=LO+rng.rand(npop,3)*(HI-LO); nf=0
    val=fitness(pair,carrier,X,metric); nf+=npop; best=X[val.argmax()].copy(); bv=val.max(); hist=[bv]; gc=None
    for gtab in range(1,gens+1):
        # fitness-proportionate selection on shifted fitness
        sh=val-val.min()+1e-9; p=sh/sh.sum(); idx=rng.choice(npop,npop,p=p); par=X[idx]
        # SBX crossover
        child=par.copy(); eta=15.0
        for k in range(0,npop-1,2):
            if rng.rand()<0.9:
                u=rng.rand(3); beta=np.where(u<=0.5,(2*u)**(1/(eta+1)),(1/(2*(1-u)))**(1/(eta+1)))
                p1,p2=par[k],par[k+1]
                child[k]=np.clip(0.5*((1+beta)*p1+(1-beta)*p2),LO,HI)
                child[k+1]=np.clip(0.5*((1-beta)*p1+(1+beta)*p2),LO,HI)
        # polynomial mutation
        mut=rng.rand(npop,3)<(1/3.); d=rng.rand(npop,3)-0.5
        child=np.clip(child+mut*d*0.1*(HI-LO),LO,HI)
        cv=fitness(pair,carrier,child,metric); nf+=npop
        # elitist replacement
        allX=np.vstack([X,child]); allv=np.concatenate([val,cv]); order=allv.argsort()[::-1][:npop]
        X=allX[order]; val=allv[order]
        if memetic:  # PSO refinement of the ne elites
            E=X[:ne].copy(); Vv=np.zeros((ne,3)); pb=E.copy(); pv=val[:ne].copy(); g=pb[pv.argmax()].copy()
            for _ in range(npso):
                r1,r2=rng.rand(ne,3),rng.rand(ne,3); Vv=0.6*Vv+1.5*r1*(pb-E)+1.5*r2*(g-E); E=np.clip(E+Vv,LO,HI)
                ev=fitness(pair,carrier,E,metric); nf+=ne; im=ev>pv; pb[im]=E[im]; pv[im]=ev[im]
                if pv.max()>g[pv.argmax():pv.argmax()+1].sum(): g=pb[pv.argmax()].copy()
            X[:ne]=pb; val[:ne]=pv
        if val.max()>bv: bv=val.max(); best=X[val.argmax()].copy()
        hist.append(bv)
        if gc is None and gtab>q and abs(hist[gtab]-hist[gtab-q])<tol: gc=gtab
    return dict(best=best, val=float(bv), gen_conv=gc if gc else gens, nfev=nf, hist=hist)

def main(seeds=10):
    os.makedirs(RESULTS,exist_ok=True)
    rows_f=[]; rows_e=[]
    for pair in PAIRS:
        for carrier in CARRIERS:
            of=dense_grid(pair,carrier,"PEC_f"); oe=dense_grid(pair,carrier,"PEC_eps")
            rows_f.append(dict(pair=pair,carrier=carrier,**{k:of[k] for k in ["w","s1","Vdot","phi","dP_kPa","PEC_f","PEC_eps"]}))
            rows_e.append(dict(pair=pair,carrier=carrier,**{k:oe[k] for k in ["w","s1","Vdot","phi","dP_kPa","PEC_f","PEC_eps"]}))
    pd.DataFrame(rows_f).to_csv(os.path.join(RESULTS,"optima_PEC_f.csv"),index=False,float_format="%.6g")
    pd.DataFrame(rows_e).to_csv(os.path.join(RESULTS,"optima_PEC_eps.csv"),index=False,float_format="%.6g")
    # algorithm comparison on a representative case per pair (EG/water), PEC_eps objective
    comp=[]
    for pair in PAIRS:
        for name,fn in [("GA",lambda s:run_ga(pair,"EG/water 60:40",seed=s)),
                        ("PSO",lambda s:run_pso(pair,"EG/water 60:40",seed=s)),
                        ("memetic_GA_PSO",lambda s:run_ga(pair,"EG/water 60:40",seed=s,memetic=True))]:
            res=[fn(s) for s in range(seeds)]
            comp.append(dict(pair=pair,algorithm=name,
                             gen_conv_mean=float(np.mean([r["gen_conv"] for r in res])),
                             nfev_mean=float(np.mean([r["nfev"] for r in res])),
                             PEC_eps_mean=float(np.mean([r["val"] for r in res]))))
    pd.DataFrame(comp).to_csv(os.path.join(RESULTS,"algorithm_comparison.csv"),index=False,float_format="%.6g")
    print("optima_PEC_f.csv, optima_PEC_eps.csv, algorithm_comparison.csv written")
    print(pd.DataFrame(rows_e)[["pair","carrier","w","Vdot","PEC_f","PEC_eps"]].to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=10); a=ap.parse_args()
    main(seeds=a.seeds)
