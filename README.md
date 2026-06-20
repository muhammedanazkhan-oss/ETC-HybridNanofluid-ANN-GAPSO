# Hybrid-nanofluid evacuated-tube collector: reduced-order solver, ANN surrogate and GA-PSO optimisation

Reproducibility package for the article

> *ANN-surrogate-driven memetic GA-PSO optimisation of the thermo-hydraulic performance evaluation
> criterion (pumping-power ratio) for a hybrid-nanofluid direct-flow evacuated-tube collector.*

[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
&nbsp;*(badge becomes active after the Zenodo deposit — see "Archiving on Zenodo" below)*

It contains everything needed to regenerate the study from scratch: the reduced-order
thermo-hydraulic solver, the dataset-generation script, the full-factorial dataset (54,432 runs),
the trained multi-output ANN surrogate and gradient-boosting benchmark, the optimisation and
sensitivity scripts, and a dynamic thermophysical-property datasheet.

---

## Repository structure

```
.
├── README.md                 # this file
├── LICENSE                   # MIT
├── CITATION.cff              # citation metadata
├── .zenodo.json              # Zenodo deposit metadata
├── requirements.txt          # Python dependencies
├── run_all.py                # one command to reproduce the whole pipeline
├── src/
│   ├── properties.py         # particle + carrier property models, design space, bounds
│   ├── solver.py             # reduced-order thermo-hydraulic solver (Eq. 1-40); PEC_f and PEC_eps
│   ├── generate_dataset.py   # builds the 54,432-run full-factorial dataset
│   ├── train_surrogate.py    # trains the ANN (predicts PEC_f, Nu, eta, Wp) + GB benchmark
│   ├── optimise.py           # GA / PSO / memetic GA-PSO / dense-grid on PEC_eps
│   ├── sensitivity.py        # BC, viscosity-closure, minor-loss, matched-pumping checks
│   └── build_datasheet.py    # builds the Excel property datasheet
├── data/
│   ├── full_factorial_dataset.csv             # 54,432 rows x 20 columns
│   └── ETC_HybridNanofluid_PropertyDatasheet.xlsx   # dynamic property datasheet
├── models/
│   ├── ann_surrogate.joblib  # trained multi-output ANN (+ scalers, metadata)
│   └── gb_benchmark.joblib   # gradient-boosting benchmark
└── results/
    ├── metrics.json                       # held-out + cross-validation accuracy
    ├── optima_PEC_f.csv                   # classical friction-factor optima (Table 10)
    ├── optima_PEC_eps.csv                 # directly re-optimised PEC_eps optima (Table 11)
    ├── optima_manuscript_reported.csv     # the values reported in the paper (for cross-reference)
    ├── algorithm_comparison.csv           # GA vs PSO vs memetic GA-PSO
    └── sensitivity.json                   # Section 4.5 model-sensitivity checks
```

## Requirements & installation

Python 3.9+ and the packages in `requirements.txt`:

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```

## Quick start (reproduce everything)

```bash
python run_all.py
```

or run the steps individually (each script is standalone):

```bash
cd src
python generate_dataset.py     # -> data/full_factorial_dataset.csv
python train_surrogate.py      # -> models/*.joblib, results/metrics.json   (set RUN_CV=1 for cross-validation)
python optimise.py --seeds 30  # -> results/optima_*.csv, algorithm_comparison.csv
python sensitivity.py          # -> results/sensitivity.json
python build_datasheet.py      # -> data/ETC_HybridNanofluid_PropertyDatasheet.xlsx
```

Determinism: a fixed seed (42) is used for the data split, network initialisation and the
optimiser. `MLP_CAP` (default 15000) caps the ANN training subsample for tractable training on a
laptop; raise or remove it to train on the full set.

## The two PEC definitions

* `PEC_f`   = (Nu/Nu_bf) / (f/f_bf)^(1/3)              — classical friction-factor **diagnostic**
* `PEC_eps` = (Nu/Nu_bf) / (Wp_hnf/Wp_bf)^(1/3)        — **primary** pumping-power-ratio metric
            = PEC_f / (rho_hnf/rho_bf)^(1/3)            at equal volumetric flow rate

The ANN is trained on the smoother diagnostic `PEC_f`; the primary metric `PEC_eps` is recovered
analytically from `PEC_f` and the (model-derived) density ratio.

## Key result

Within this reduced-order model and the pumping-power-ratio metric, **no hybrid nanofluid produces
a robust improvement over its base fluid**. The dense copper of Al2O3-Cu raises the pumping power,
and the (rho_hnf/rho_bf)^(1/3) factor cancels almost the entire friction-factor-form gain, so the
best case — **Al2O3-Cu / EG-water** — only reaches near parity. It is a candidate for experimental
validation, not a demonstrated performance gain. See `results/optima_manuscript_reported.csv`.

## Consistency with the manuscript (please read)

This is a clean **reference implementation**. It reproduces the full methodology and the central
qualitative findings. A few quantitative notes:

* **Geometry / hydraulics / water & oil PEC** reproduce the manuscript closely (e.g. wetted area
  A_s = 0.0452 m^2; maximum optimum dP ~ 12.6 kPa; Al2O3-Cu water/oil PEC_f to ~3 decimals).
* **EG-water optimum** sits exactly on the laminar-turbulent transition (Re ~ 2800). There, the
  absolute PEC is highly sensitive to the viscosity closure and to the transition treatment — this
  is the very sensitivity quantified in Section 4.5. The manuscript's headline EG-water value uses
  the authors' original calibration; this reference solver returns a slightly higher value at the
  transition. The directly re-optimised values reported in the paper are provided in
  `results/optima_manuscript_reported.csv` for cross-reference.
* **Surrogate**: the manuscript's headline metrics correspond to a Keras shared-trunk network;
  this package provides an scikit-learn `MLPRegressor` reference surrogate that predicts `PEC_f`
  accurately (see `results/metrics.json`). Exact metrics differ slightly between the two
  implementations, as expected.

The intended use of this archive is methodological reproducibility; absolute PEC values inherit the
model assumptions discussed in the paper.

## Dataset columns (`data/full_factorial_dataset.csv`)

`pair, carrier, w_pct, s1_pct, Vdot_lpm, Ti_C, Is_Wm2, Ta_C` (inputs) and
`phi_hnf, Re, Pr, Nu, f, dP_kPa, Wp_W, eta, rho_hnf, rho_bf, PEC_f, PEC_eps` (outputs).

## Using the trained surrogate

```python
import joblib, numpy as np
m = joblib.load("models/ann_surrogate.joblib")
# features: one-hot pair (3) + one-hot carrier (3) + [w_pct, s1_pct, Vdot_lpm, Ti_C, Is_Wm2, Ta_C]
# order of one-hots: m["pairs"], m["carriers"]
x = np.array([[1,0,0, 0,1,0, 3.0, 25, 1.71, 60, 700, 25]])   # Al2O3-Cu, EG/water
pred = m["y_scaler"].inverse_transform(m["model"].predict(m["x_scaler"].transform(x)))
PEC_f, Nu, eta, logWp = pred[0]; Wp = np.exp(logWp)
```

## Archiving on Zenodo and getting a DOI

1. Create a new GitHub repository and push this folder (or upload the zip and let GitHub unpack it).
2. Go to <https://zenodo.org> → log in → **Settings → GitHub**, and toggle the repository **On**.
3. On GitHub, create a **release** (e.g. tag `v1.0.0`). Zenodo automatically archives the release
   and mints a DOI.
4. Copy the DOI badge from Zenodo into the badge line at the top of this README, and add the DOI to
   the manuscript's "Data and code availability" statement and to `CITATION.cff` / `.zenodo.json`.
5. (Optional) add your ORCID in `CITATION.cff` and `.zenodo.json` before the release.

## License & citation

Released under the MIT License (see `LICENSE`). If you use this software or dataset, please cite the
associated article and this archive (see `CITATION.cff`).
