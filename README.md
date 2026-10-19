# Time Series Forecasting with LSTM and Feature Engineering

**Victoria University — NIT3251 Advanced Data Science, Assessment 2.**

This research evaluates next-trading-session NVIDIA (NVDA) closing-price forecasts using naïve persistence, ARIMA(2,1,2), and three residual LSTM variants:

- **A — price-only:** NVIDIA Close (one feature; primary LSTM).
- **B — technical indicators:** Close plus seven SMA, EMA, RSI and MACD-derived features (eight features).
- **C — market context:** Model B plus SPY and XLK one-session log returns (ten features).

The target is Yahoo's retrospectively split-adjusted **Close in USD**, not dividend-adjusted Adj Close. The study uses 30-session windows and chronological development training (2015–2022), validation (2023–2024), frozen final refitting (2015–2024) and independent testing (2025). During testing, LSTM weights/scalers and ARIMA coefficients remain fixed; observed prices update the information available for subsequent one-step forecasts.

## Principal findings

The independent evaluation covers **250 sessions and 2,750 forecasts**. LSTM family values are means of three seed-specific metrics, not prediction ensembles.

| Model | Test MAE (USD) | Test RMSE (USD) |
|---|---:|---:|
| Naïve persistence | 3.154681 | 4.290545 |
| ARIMA(2,1,2) | **3.110051** | **4.239547** |
| A — price-only LSTM | 3.187466 | 4.310978 |
| B — technical indicators | 3.181817 | 4.325776 |
| C — indicators + context | 3.179046 | 4.315488 |

ARIMA achieved the lowest observed errors, with a 1.415% MAE improvement over naïve. No LSTM family outperformed naïve in mean MAE or RMSE. Feature effects were small and inconsistent across metrics, seeds and development/test periods. All four primary paired bootstrap intervals include zero; the results do not establish statistically reliable superiority or profitable trading. See [Notebook 06](notebooks/06_final_evaluation.ipynb) for the final interpretation.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/` | Six executed research notebooks; analysis, demonstrations and conclusions |
| `src/market_forecasting/` | Authoritative data, feature, model, preprocessing and evaluation functions |
| `configs/` | Temporal boundaries, selected settings and frozen experiment protocols |
| `scripts/` | Acquisition, checks and experiment orchestration |
| `tests/` | Synthetic tests of calculations, alignment, causality and frozen inference |
| `docs/` | Data conventions, methodological decisions and development-result summaries |
| `data/raw/` | Three original Yahoo snapshots; **not tracked** |
| `artifacts/` | Saved predictions, histories, models and final manifests; **not tracked** |

## Data Availability and Reproducibility

This repository contains the complete Python implementation, configurations,
tests, research documentation, and six executed Jupyter notebooks.

The project uses historical Yahoo Finance data for NVIDIA (NVDA), SPY, and XLK,
covering January 2015 to December 2025. The original datasets and saved
experiment artifacts are not included in the repository because redistribution
permission has not been established.

The executed notebooks can be inspected directly on GitHub. Re-executing the
analyses and verifying the saved models requires access to the original
datasets and experiment artifacts.

### Required Datasets

The research uses three frozen data snapshots:

| Dataset | File |
|---|---|
| NVIDIA | `data/raw/NVDA_yahoo_snapshot.json` |
| S&P 500 ETF | `data/raw/SPY_yahoo_snapshot.json` |
| Technology Sector ETF | `data/raw/XLK_yahoo_snapshot.json` |

Each snapshot contains 2,766 trading sessions, from 2 January 2015 to
31 December 2025.

Dataset specifications, original checksums, and preprocessing conventions are
documented in [Data Conventions](docs/DATA_CONVENTIONS.md) and
[Data Verification](docs/data_verification.json).

### Saved Experiment Artifacts

The original experiments produced saved predictions, training histories,
model checkpoints, evaluation results, and verification records.

These are organised under the following directories:

- `artifacts/naive_v1/`
- `artifacts/arima_v1/`
- `artifacts/lstm_v1/`
- `artifacts/features_v1/`
- `artifacts/final_training_v1/`
- `artifacts/final_evaluation_v1/`

These artifacts are excluded from GitHub. When available through an authorised
access arrangement, they must be restored to their original directory
locations to reproduce the recorded results.

The final evaluation notebook uses saved forecasts rather than generating
new test predictions. Full artifact verification additionally requires the
original development records and frozen model files.

The installation, notebook execution, and integrity-check instructions are
provided below.

## Installation

The recorded environment is **Python 3.12.14 on macOS ARM64**. The full lockfile was installed in a clean Python 3.12 environment during submission verification. Other platforms are unverified; the lock includes platform-specific packages such as `appnope`. Python metadata allowing 3.11–3.12 is not evidence of verification on every version.

From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lstm-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m pytest -q
```

The lock includes TensorFlow 2.21.0, Keras 3.15.1, statsmodels 0.15.0 and the Jupyter dependencies. Earlier milestone lockfiles document historical environments; do not install all four together. No API keys, credentials or custom environment variables are required to inspect saved evidence or run the tests.

## Running the notebooks

After supplying the authorised prerequisites:

```sh
python scripts/check_snapshot.py
python -m ipykernel install --user --name nit3251 --display-name "NIT3251 (Python 3.12)"
jupyter lab
```

Select the **NIT3251 (Python 3.12)** kernel. Start Jupyter from the repository root and run notebooks in this order; their path discovery also supports the `notebooks/` directory.

| Notebook | Additional local inputs | Execution behaviour |
|---|---|---|
| [01 — Data understanding](notebooks/01_data_understanding.ipynb) | All three snapshots | Training-period EDA and quality checks |
| [02 — Dataset and naïve](notebooks/02_dataset_and_naive_baseline.ipynb) | NVDA; naïve summary, metadata and predictions | Reconstructs windows and baseline in memory |
| [03 — ARIMA](notebooks/03_arima_benchmark.ipynb) | NVDA; ARIMA and naïve records | **Refits the selected development ARIMA** and replays development forecasts; no new order search |
| [04 — LSTM](notebooks/04_lstm_experiments.ipynb) | NVDA; all `lstm_v1` runs; baseline records | Loads all 15 checkpoints and verifies development predictions; no training |
| [05 — Features](notebooks/05_feature_experiments.ipynb) | All snapshots; `features_v1`; baseline records; Milestone 5 seed-42 final summary | Verifies features and nine saved checkpoints; no training |
| [06 — Final evaluation](notebooks/06_final_evaluation.ipynb) | All final-training files; final evaluation JSON/source archive | Recalculates metrics and recorded uncertainty from saved forecasts; no new test forecasts |

For inspection **without any refitting**, read Notebook 03's existing executed output rather than running all its cells. No notebook requires a machine-specific absolute path. Floating-point agreement across hardware/software is subject to the recorded tolerances; deterministic settings do not guarantee bit-identical retraining.

## Integrity checks and rebuilding

With the complete authorised evidence present:

```sh
python scripts/check_snapshot.py
python scripts/run_final_training.py verify
```

`verify` reloads final models, checks scalers and hashes, and performs pre-test inference checks without fitting or repeating the independent evaluation. Its manifest protects exact versions of selected documentation and locked notebooks as well as artifacts. A mismatch must be investigated, not bypassed by editing hashes. The original training/evaluation `source.zip` archives capture the implementation actually executed; Git revision identifiers alone are insufficient provenance.

Inspect results through the notebooks and saved JSON first. Rebuilding experiments is a **separate, explicitly planned operation in an isolated working copy**, with new output/provenance destinations and the original evidence preserved. The existing entry points are `build_naive_baseline.py`, `run_arima_benchmark.py`, `run_lstm_experiments.py` (`selection`, then `final`) and `run_feature_experiments.py`. Their configurations and reproduction details are documented in [Milestone 4](docs/MILESTONE_4.md), [Milestone 5](docs/MILESTONE_5.md) and [Milestone 6](docs/MILESTONE_6.md). Several runners/reporters use fixed destinations or write documentation; they are not read-only verification commands. A new final fit creates new artifact identities and must not replace the frozen manifest. Do not rerun `run_final_training.py evaluate` to recreate or overwrite the completed 2025 evaluation.

## Supporting methodology

- [Data definitions and temporal boundaries](docs/DATA_CONVENTIONS.md)
- [Research decisions](docs/DECISIONS.md)
- [Progress and verification evidence](docs/PROGRESS.md)
- [Research roadmap](docs/PLAN.md)
