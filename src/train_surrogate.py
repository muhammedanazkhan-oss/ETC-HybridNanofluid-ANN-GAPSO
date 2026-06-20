"""
Train the multi-output ANN surrogate and the gradient-boosting benchmark.

Inputs (12): one-hot hybrid pair (3) + one-hot carrier (3) + six continuous controls
             (w_pct, s1_pct, Vdot_lpm, Ti_C, Is_Wm2, Ta_C).
ANN target : the classical friction-factor diagnostic PEC_f and the auxiliary outputs Nu, eta, Wp.
             The primary pumping-power-ratio metric is recovered analytically as
             PEC_eps = PEC_f / (rho_hnf/rho_bf)^(1/3).
Saves models/ann_surrogate.joblib, models/gb_benchmark.joblib, results/metrics.json.
"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "full_factorial_dataset.csv")
MODELS = os.path.join(HERE, "..", "models")
RESULTS = os.path.join(HERE, "..", "results")
SEED = 42
TARGETS = ["PEC_f", "Nu", "eta", "Wp_W"]

def features(df):
    pairs = sorted(df["pair"].unique()); carrs = sorted(df["carrier"].unique())
    oh_p = np.array([[1.0 if p==c else 0.0 for c in pairs] for p in df["pair"]])
    oh_c = np.array([[1.0 if p==c else 0.0 for c in carrs] for p in df["carrier"]])
    cont = df[["w_pct","s1_pct","Vdot_lpm","Ti_C","Is_Wm2","Ta_C"]].to_numpy(float)
    return np.hstack([oh_p, oh_c, cont]), pairs, carrs

def metrics(y, yp):
    return dict(R2=float(r2_score(y,yp)), RMSE=float(np.sqrt(mean_squared_error(y,yp))),
                MAE=float(mean_absolute_error(y,yp)),
                MAPE=float(np.mean(np.abs((y-yp)/np.where(np.abs(y)>1e-9,y,1e-9)))*100))

def main():
    os.makedirs(MODELS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_csv(DATA)
    X, pairs, carrs = features(df)
    Y = df[TARGETS].to_numpy(float).copy()
    Y[:,3] = np.log(np.maximum(Y[:,3], 1e-9))          # Wp in log space
    Xtr,Xte,Ytr,Yte = train_test_split(X, Y, test_size=0.15, random_state=SEED)
    xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(Ytr)
    ann = MLPRegressor(hidden_layer_sizes=(96,96,48), activation="relu", alpha=1e-6,
                       batch_size=512, learning_rate_init=1.2e-3, max_iter=300,
                       early_stopping=True, n_iter_no_change=25, random_state=SEED)
    ann.fit(xs.transform(Xtr), ys.transform(Ytr))
    Pann = ys.inverse_transform(ann.predict(xs.transform(Xte)))
    gb = {}; Pgb = np.zeros_like(Yte)
    for j,t in enumerate(TARGETS):
        g = HistGradientBoostingRegressor(max_iter=250, max_depth=7, learning_rate=0.05, random_state=SEED)
        g.fit(Xtr, Ytr[:,j]); gb[t]=g; Pgb[:,j]=g.predict(Xte)
    # de-log Wp for reporting
    def delog(M): M=M.copy(); M[:,3]=np.exp(M[:,3]); return M
    Yte_r, Pann_r, Pgb_r = delog(Yte), delog(Pann), delog(Pgb)
    rep = {"n_test": int(len(Yte)), "n_train": int(len(Ytr)), "inputs": int(X.shape[1]),
           "ANN": {}, "GB": {}}
    for j,t in enumerate(TARGETS):
        rep["ANN"][t] = metrics(Yte_r[:,j], Pann_r[:,j])
        rep["GB"][t]  = metrics(Yte_r[:,j], Pgb_r[:,j])
    if os.environ.get('RUN_CV','0')=='1':
      # 5-fold CV of the ANN on PEC_f
      kf = KFold(5, shuffle=True, random_state=SEED); r2s=[]; rmses=[]; mapes=[]
      yidx = TARGETS.index("PEC_f")
      for tr,va in kf.split(X):
          xs2 = StandardScaler().fit(X[tr]); ys2 = StandardScaler().fit(Y[tr])
          m = MLPRegressor(hidden_layer_sizes=(96,96,48), alpha=1e-6, learning_rate_init=1.2e-3,
                           max_iter=200, early_stopping=True, random_state=SEED).fit(xs2.transform(X[tr]), ys2.transform(Y[tr]))
          p = ys2.inverse_transform(m.predict(xs2.transform(X[va])))[:,yidx]; yv=Y[va][:,yidx]
          r2s.append(r2_score(yv,p)); rmses.append(np.sqrt(mean_squared_error(yv,p)))
          mapes.append(np.mean(np.abs((yv-p)/yv))*100)
      rep["CV_PEC_f"] = dict(R2_mean=float(np.mean(r2s)), R2_sd=float(np.std(r2s)),
                             RMSE_mean=float(np.mean(rmses)), MAPE_mean=float(np.mean(mapes)), MAPE_sd=float(np.std(mapes)))
    joblib.dump({"model":ann,"x_scaler":xs,"y_scaler":ys,"targets":TARGETS,
                 "pairs":pairs,"carriers":carrs,"wp_log":True}, os.path.join(MODELS,"ann_surrogate.joblib"))
    joblib.dump({"models":gb,"targets":TARGETS,"pairs":pairs,"carriers":carrs,"wp_log":True},
                os.path.join(MODELS,"gb_benchmark.joblib"))
    json.dump(rep, open(os.path.join(RESULTS,"metrics.json"),"w"), indent=2)
    print("ANN PEC_f:", rep["ANN"]["PEC_f"]); print("GB  PEC_f:", rep["GB"]["PEC_f"])
    print("CV PEC_f:", rep["CV_PEC_f"]); print("saved models + results/metrics.json")

if __name__ == "__main__":
    main()
