"""
Generate the full-factorial dataset (54,432 runs) by evaluating the reduced-order solver on the
design grid of properties.DESIGN. Writes data/full_factorial_dataset.csv.

Columns: pair, carrier, w_pct, s1_pct, Vdot_lpm, Ti_C, Is_Wm2, Ta_C,
         phi_hnf, Re, Pr, Nu, f, dP_kPa, Wp_W, eta, rho_hnf, rho_bf, PEC_f, PEC_eps
"""
import os, itertools, numpy as np, pandas as pd
from properties import PAIRS, CARRIERS, DESIGN
from solver import solve

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "full_factorial_dataset.csv")

def main():
    grid = list(itertools.product(DESIGN["w_pct"], DESIGN["s1_pct"], DESIGN["Vdot_lpm"],
                                  DESIGN["Ti_C"], DESIGN["Is_Wm2"], DESIGN["Ta_C"]))
    arr = np.array(grid, float)
    rows = []
    for pair in PAIRS:
        for carrier in CARRIERS:
            r = solve(pair, carrier, arr[:,0]/100.0, arr[:,1]/100.0, arr[:,2],
                      arr[:,3], arr[:,4], arr[:,5])
            block = np.column_stack([arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5],
                                     r["phi"], r["Re"], r["Pr"], r["Nu"], r["f"], r["dP_kPa"],
                                     r["Wp"], r["eta"], r["rho"], r["rho_bf"], r["PEC_f"], r["PEC_eps"]])
            for b in block:
                rows.append([pair, carrier] + list(b))
    cols = ["pair","carrier","w_pct","s1_pct","Vdot_lpm","Ti_C","Is_Wm2","Ta_C",
            "phi_hnf","Re","Pr","Nu","f","dP_kPa","Wp_W","eta","rho_hnf","rho_bf","PEC_f","PEC_eps"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUT, index=False, float_format="%.6g")
    print(f"wrote {len(df)} rows -> {os.path.relpath(OUT)}")
    return df

if __name__ == "__main__":
    main()
