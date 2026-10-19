# Market Forecasting with LSTM

An individual NIT3251 Advanced Data Science research project investigating next-session stock-price forecasting. NVIDIA (NVDA) is the initial case study; SPY and XLK are candidate market and sector context series.

## Research objectives
Compare a last-observed-price baseline and a traditional statistical model with LSTMs using stock history, technical features and market context. Evaluate whether additional inputs improve validation performance under a consistent forecast-origin protocol, followed by a reserved final test.

The target is Yahoo split-adjusted Close in USD. Volume features remain deferred pending verification of adjustment semantics. Price-only LSTM tuning and the nine-run controlled feature comparison are complete. See [data conventions](docs/DATA_CONVENTIONS.md).

## Current implementation
Frozen data, calendar auditing, price-only development windows and the naive validation baseline are implemented. ARIMA and the residual price-only LSTM are implemented. Fifteen LSTM fits selected configuration C; its mean validation MAE is 1.705023 USD, compared with 1.689494 USD for the naive benchmark. The subsequent common-warm-up feature comparison found no average improvement over price-only LSTM or naive; see [Milestone 6](docs/MILESTONE_6.md). Independent 2025 evaluation is complete. ARIMA has the lowest observed errors, while none of the LSTM families improves on naïve mean MAE or RMSE; the primary uncertainty intervals include zero. [Notebook 06](notebooks/06_final_evaluation.ipynb) presents the final evidence and conclusions. See the [LSTM results](docs/MILESTONE_5.md). See [implementation and verification status](docs/PROGRESS.md), the [roadmap](docs/PLAN.md) and the [benchmark summary](docs/MILESTONE_4.md).

## Frozen final-training models
The final specification uses naive persistence, ARIMA(2,1,2), primary price-only LSTM A, technical-indicator LSTM B and market-context LSTM C. One ARIMA and nine fresh LSTM fits were completed and frozen on 18 October 2026, using only 2015–2024 observations. All three LSTM variants share 2,437 fitting targets and fixed 19-epoch training. The separate authorised independent evaluation preserves these fitted models unchanged.

The [final protocol](configs/final_training_v1.json) records the approved settings and future comparisons. The local [freeze manifest](artifacts/final_training_v1/manifest.json) binds model identities, configurations, scalers, source/data hashes, package versions and independently verified artifacts. See [final-stage progress](docs/PROGRESS.md#7-final-training-and-independent-evaluation) for verification evidence and limitations.

```sh
python scripts/run_final_training.py verify
```

Verification reloads the saved artifacts and uses only pre-test examples. The `fit` phase performs the single approved final-training experiment and refuses an existing destination; it is not a routine verification command. Milestone 8A explicitly authorised both the code-level release gate and the [evaluation configuration](configs/final_evaluation_v1.json). The single independent evaluation is complete: 250 sessions and 2,750 forecasts, with full-precision evidence in [evaluation.json](artifacts/final_evaluation_v1/evaluation.json). The `evaluate` phase refuses the existing destination and must not be rerun to replace results. [Notebook 06](notebooks/06_final_evaluation.ipynb) verifies and interprets the saved results without repeating model inference or fitting.

Final checkpoints, preprocessing and `source.zip` remain in `artifacts/final_training_v1/`, excluded from Git. Preserve this directory and the original snapshots for exact artifact reproduction. The source archive records the actual implementation used for fitting, including uncommitted code; the Git revision alone does not identify that implementation. Repeated inference is verified in the recorded environment; deterministic training settings do not guarantee bitwise agreement across different software or hardware.

The `dry-run` phase exercises all frozen models, chronological observation updates, metrics and bootstrap calculations on the final 30 permitted pre-test sessions:

```sh
python scripts/run_final_training.py dry-run > /tmp/nit3251-pretest-check.json
```

This temporary record is a machinery check, not independent accuracy evidence: the models were fitted through the demonstrated period. It does not modify frozen artifacts or produce an evaluation directory. Reusable plotting functions in `market_forecasting.evaluation_plots` return figures without exporting files. The authorised evaluation preserves one `evaluation.json` record, its exact evaluation-source archive and eight verified diagnostic figures in `artifacts/final_evaluation_v1/`. These local artifacts are excluded from Git and must be preserved alongside the frozen training artifacts.

## Installation
Python 3.12 is the verified environment; package metadata permits Python 3.11–3.12, but that does not establish verification on both versions. Run from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lstm-lock.txt
python -m pip install -e .
python -m pip check
```

The milestone lockfile records the current verified development dependencies. `requirements-foundation-lock.txt` preserves the earlier foundation environment. The current environment includes statsmodels 0.15.0, TensorFlow 2.21.0 and Keras 3.15.1. Dashboard dependencies remain deferred. Earlier milestone lockfiles are preserved.

## Data prerequisites and reproducibility
Exact reproduction requires the original local Yahoo JSON snapshots under `data/raw/`. These files are excluded from Git. [The acquisition record](docs/data_verification.json) contains retrieval metadata and checksums for NVDA, SPY and XLK: 2,766 observations each, retrieved on 11 October 2026 for 2015–2025.

A new clone alone cannot reproduce the recorded numerical results without those snapshots. A later vendor download may contain revised history and must be treated as a new dataset version. The acquisition script documents the original procedure, but currently writes a fixed audit path; do not rerun it against preserved evidence. A refresh requires separately versioned data and audit destinations. Data access does not confer redistribution rights; permitted submission and sharing arrangements remain to be established.

With the original snapshots available:

```sh
python scripts/check_snapshot.py
python -m pytest -q
jupyter lab
```

Notebooks are the primary research narrative and demonstration interface; reusable Python modules implement data processing, features, models and evaluation. Each notebook calls the actual implementation, examines its behaviour and uses evidence to assess whether changes are warranted. Keep one authoritative notebook per stage; do not duplicate production algorithms or generate separate executed notebook copies.

Run [notebook 01](notebooks/01_data_understanding.ipynb) for training-period EDA and data-quality assessment. Its outputs remain inline in that notebook, and its loader decodes only training-period price values. [Notebook 02](notebooks/02_dataset_and_naive_baseline.ipynb) covers development windows and validation metrics. Existing historical experiment artifacts remain preserved.

The calendar script displays a training-only, timestamp-only audit without writing a report. The preserved historical calendar record remains unchanged. Baseline scripts write fixed, version-labelled destinations and refuse to overwrite them. Notebook 02 recomputes baseline results in memory. See the [benchmark reproduction notes](docs/MILESTONE_4.md) before running artifact-producing scripts.

## Repository structure
| Path | Responsibility |
|---|---|
| `configs/` | Forecast configuration and temporal boundaries |
| `src/market_forecasting/` | Reusable loading, validation, calendar and dataset logic |
| `scripts/` | Acquisition, integrity checks and experiment artifact generation |
| `notebooks/` | Primary research narrative, implementation demonstrations and critical analysis |
| `tests/` | Automated data-integrity, alignment and evaluation checks |
| `docs/` | Roadmap, progress, decisions, conventions and technical evidence |
| `data/` | Local vendor data, excluded from Git |
| `artifacts/` | Generated outputs, excluded from Git |

## Supporting documentation
- [Roadmap and completion criteria](docs/PLAN.md)
- [Progress and verification](docs/PROGRESS.md)
- [Research and engineering decisions](docs/DECISIONS.md)
- [Data conventions and limitations](docs/DATA_CONVENTIONS.md)
- [Milestone 2: environment and audit](docs/MILESTONE_2.md)
- [Milestone 3: dataset construction](docs/MILESTONE_3.md)
- [Milestone 4: forecasting benchmarks](docs/MILESTONE_4.md)
- [Milestone 5: residual LSTM experiments and reproduction](docs/MILESTONE_5.md)
- [Milestone 6: controlled feature experiments](docs/MILESTONE_6.md)
- [Notebook 06: independent evaluation and research conclusions](notebooks/06_final_evaluation.ipynb)
- [Notebook 05: feature analysis and comparisons](notebooks/05_feature_experiments.ipynb)
- [Notebook 04: saved LSTM experiment analysis](notebooks/04_lstm_experiments.ipynb)

A Streamlit/Plotly demonstration, final report and assessment defence are planned after the forecasting experiments. The demonstration will present saved results; live trading and profitability claims are outside scope.

## Statistical benchmark and feature experiments
See [Milestone 4](docs/MILESTONE_4.md) for the ARIMA protocol, executed comparison and reproduction command. [Notebook 03](notebooks/03_arima_benchmark.ipynb) displays saved forecasts. Technical feature engineering remains central: SMA, EMA, RSI and MACD candidates and three controlled LSTM variants are documented in the [proposal](docs/FEATURE_EXPERIMENT_PROPOSAL.md), now implemented and evaluated in [Milestone 6](docs/MILESTONE_6.md).

## Code documentation standard
Python modules and functions generally use one-sentence docstrings; a second sentence is added only when needed to clarify a non-obvious contract. Repetitive Parameters, Returns, Raises and Notes sections are avoided. Concise comments beside the implementation explain methodological safeguards, important shapes, transformations and non-obvious calculations. Comments explain why an operation is needed without narrating obvious statements or claiming unimplemented capabilities.

This standard applies to subsequent milestones, including scaling, model construction, optimiser settings, seed handling, checkpoint restoration, inverse transformation and evaluation. Reusable logic remains in Python modules; notebooks demonstrate and interpret experiments. Documentation changes are verified without unrelated refactoring. See the [Milestone 5 design](docs/MILESTONE_5_DESIGN.md) for the bounded LSTM protocol.
