# Laptop Price Intelligence & Prediction

[Live Streamlit application](https://laptop-price-intelligence-haitham.streamlit.app/) | [GitHub repository](https://github.com/haithamalfraidi-eng/laptop-price-intelligence)

A Python analytics and regression portfolio project with a Streamlit dashboard. It describes the supplied dataset rather than current market prices or purchase value.

## Features

- Market analysis: manufacturer/category filters, price distributions, RAM relationships, and business observations computed from filtered records. Counts accompany category medians; dataset coverage is not market share.
- Specification comparison: category, RAM, exact CPU text, SSD/HDD capacities and total storage. At least three matching peers are needed for a median and relative price difference. Approximate alternatives show matching flags and specification differences; they are not quality rankings.
- Manual price prediction: manufacturer, category, CPU brand, OS, RAM, screen size, weight, SSD, HDD and total storage. Validation rejects invalid/nonfinite inputs and warns outside training ranges. Total storage may include flash/hybrid capacity beyond SSD/HDD.
- Model performance: MAE, RMSE, R-squared and holdout-size cards, training-only cross-validation with fold variation, actual/predicted and residual charts, and the original/updated comparison.
- Existing-device predictions remain available and are explicitly labeled as potentially in-sample.

## Setup and reproduce

Validated with Python **3.11.9** on Windows. Use Python 3.11 for the closest reproduction.

```powershell
cd Laptop_Price_Intelligence_Complete
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python run_pipeline.py
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. The lock file pins the dependency closure of the validated environment; OS-specific wheel availability can differ. `requirements-ranges.txt` provides supported dependency ranges when exact versions are unavailable, but numerical identity is not guaranteed with other versions. No extra application libraries were introduced.

The included, attributed raw CSV must remain at `data/raw/laptops_test.csv`. The pipeline creates missing output directories, runs independently of the caller's working directory, and regenerates cleaned CSV, SQLite database, charts, SQL analysis, model and evaluation reports. Models/databases are ignored by Git and must be rebuilt after cloning. Load only trusted local joblib artifacts.

`reports/reproducibility.json` records Python/package versions, seed 42 and raw-file SHA-256. `reports/split_manifest.csv` records group assignments and row IDs; `holdout_predictions.csv` makes reported errors auditable. Row IDs refer to the deterministic cleaned CSV. `training_reference.csv` contains the training-only reference used by manual validation. Small floating-point differences across platforms remain possible.

## Measured evaluation results

All errors below are in **unknown dataset price units**. R-squared is not percentage accuracy.

| Evaluation | Selected model | Train/test rows | MAE | RMSE | R-squared |
|---|---|---|---:|---:|---:|
| Original recorded row split | Random Forest | 237 / 60 | 2,664,197.30 | 4,150,024.53 | 0.537763 |
| Updated identical row split | Random Forest | 237 / 60 | 2,664,197.30 | 4,150,024.53 | 0.537763 |
| Updated grouped holdout (application model) | Ridge Regression | 241 / 56 | 2,569,104.23 | 4,452,302.07 | 0.552214 |

The first two evaluations use the same data, features, candidates, random seed, 80/20 row split and training-only five-fold CV. They reproduce identically: **no same-split accuracy improvement is claimed**. The original split has 18 manufacturer/model groups on both sides.

There are 188 distinct normalized manufacturer/model groups among 297 cleaned rows. The deployed evaluation holds out 20% of groups (not exactly 20% of rows) using GroupShuffleSplit. GroupKFold keeps groups separate within training-only five-fold model selection. The grouped holdout has zero exact manufacturer/model overlap. This evaluates a different task on different devices; its metrics must not be interpreted as a direct improvement over the row split.

Grouped training CV MAE (mean +/- fold SD): median baseline **4,668,295.75 +/- 499,037.90**, Ridge **2,648,709.41 +/- 380,973.42**, Random Forest **2,664,809.40 +/- 348,781.71**. Minimum training CV MAE selects Ridge. Imputation, scaling and one-hot encoding are fitted inside each fold's pipeline. No price or model-name field is a model feature. The selected model is fitted only on training rows and evaluated once on its holdout; holdout scores do not choose the model. Baseline holdout scores are not used for selection. No hyperparameter tuning was performed against the holdout.

## Preservation and structure

- `original_lap.py` and `data/raw/laptops_test.csv`: preserved unchanged.
- `archive/original/`: local-only pre-update snapshots, excluded from Git. `original_lap.py` remains published unchanged. The recorded original evaluation is preserved in `evaluation/original_model_metrics.json`.
- `src/data_cleaning.py`: schema checks, deterministic duplicate removal and feature conversion; 2TB means 2000GB.
- `src/analytics.py`, `src/sql_analysis.py`, `sql/analysis.sql`: existing chart and SQL outputs.
- `src/pricing_engine.py`: strict peer comparisons and approximate alternatives.
- `src/prediction.py`: manual-input validation.
- `src/train_model.py`: comparable original procedure, grouped evaluation and reproducibility reports.
- `app.py`: four dashboard sections; `run_pipeline.py`: batch entry point.
- `tests/`: original checks plus validation, comparison, leakage-boundary and Streamlit workflow tests.

## Data and interpretation limitations

The owner-provided CSV has 325 rows, 13 original columns and 28 exact duplicates; cleaning retains 297 records across 12 manufacturers. One weight is unparseable and imputed within training folds. The matching upstream CSV is published by Ammok on Hugging Face with an Apache-2.0 declaration. File contents were verified identical after line-ending normalization. See [DATA_SOURCES.md](DATA_SOURCES.md) and `licenses/` for revision links, hashes, attribution and the license. This verifies the upstream declaration, not original ownership. The original collection date and currency remain unknown.

Small category and specification groups make medians and model scores unstable. Group names are normalized only for case/whitespace; aliases and related product families can still cross boundaries. This reduces one leakage risk rather than proving complete independence. The raw file's `test` name does not establish an untouched external validation set; the project creates its own split.

The model uses CPU brand rather than exact CPU performance and omits GPU, resolution, condition, geography and price date. Comparisons use exact CPU text but still omit meaningful hardware/condition differences. Flash/hybrid media with the same totals may be ambiguous. Range checks cannot detect every unsupported combination. No calibrated prediction intervals are provided; MAE is not an individual-device uncertainty bound. There is no temporal/external validation, currency conversion, live pricing, demand forecast, profit estimate or guarantee of generalization.

## Verification

Run `python run_pipeline.py` followed by `python -m unittest discover -s tests -v`. The suite exercises cleaning, storage conversion, strict/sparse peers, invalid manual inputs, disjoint holdout groups, Streamlit form submission, empty filters and missing dataset handling. Streamlit AppTest checks application execution and widgets; it does not replace a visual browser review. A headless HTTP health check was also performed during development.

Validation completed: full pipeline, 14 passing automated tests, and headless Streamlit health/page endpoints both HTTP 200. Local and deployed Chrome browser reviews were performed.

## Community Cloud deployment

Use repository `laptop-price-intelligence`, branch `main`, entry point **`streamlit_app.py`**, and Python **3.11**. `requirements.txt` delegates to the validated lock file. The entry point builds cleaned data, reports and the model once per server process, so model binaries and processed records do not need to be committed. Launch locally in cloud mode with `python -m streamlit run streamlit_app.py`.

Generated artifacts, local archives, credentials, `.env` and `.streamlit/secrets.toml` are excluded by `.gitignore`. The raw dataset is included based on the matching upstream Apache-2.0 declaration; see `DATA_SOURCES.md`. No secrets are required by this application.

`ci-template.yml` contains a GitHub Actions workflow that installs dependencies, executes the pipeline and runs all automated tests on Python 3.11. To enable it, move it to `.github/workflows/ci.yml` using an account authorized to manage workflows. It is published as a template because the publishing authorization does not include the workflow scope. `evaluation/original_model_metrics.json` is the immutable original comparison input; reports are regenerated during startup and CI.

Published on Streamlit Community Cloud at https://laptop-price-intelligence-haitham.streamlit.app/ using branch `main`, entry point `streamlit_app.py` and Python 3.11. On 2026-10-19, both public URLs returned HTTP 200 using an unauthenticated cookie-aware client. A fresh Chrome profile loaded the app without sign-in; all four tabs, manual prediction submission, expected evaluation metrics and three performance charts were checked. Screenshots of the deployed dashboard were visually reviewed. No application exceptions were observed in these checks. The Actions configuration remains a template (`ci-template.yml`), not an active CI workflow.
