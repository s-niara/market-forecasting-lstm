# Market Forecasting with LSTM

An individual NIT3251 Advanced Data Science research project investigating next-session stock-price forecasting. NVIDIA (NVDA) is the initial case study; SPY and XLK are candidate market and sector context series.

## Research objectives
Compare a last-observed-price baseline and a traditional statistical model with LSTMs using stock history, technical features and market context. Evaluate whether additional inputs improve validation performance under a consistent forecast-origin protocol, followed by a reserved final test.

The target is Yahoo split-adjusted Close in USD. Volume features remain deferred pending verification of adjustment semantics. Feature definitions and model tuning are proposed work, not completed experiments. See [data conventions](docs/DATA_CONVENTIONS.md).

## Current implementation
Frozen data, calendar auditing, price-only development windows and the naive validation baseline are implemented. No learned forecasting model has been trained and no 2025 forecast performance has been evaluated. See [implementation and verification status](docs/PROGRESS.md), the [roadmap](docs/PLAN.md) and the [first benchmark summary](docs/MILESTONE_2.md).

## Installation
Python 3.12 is the verified environment; package metadata permits Python 3.11–3.12, but that does not establish verification on both versions. Run from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-milestone2-lock.txt
python -m pip install -e .
python -m pip check
```

The milestone lockfile records the current verified development dependencies. `requirements-foundation-lock.txt` preserves the earlier foundation environment. TensorFlow, statistical modelling and dashboard dependencies will be verified at their implementation stages; installation of the current environment does not imply those stages are complete.

## Data prerequisites and reproducibility
Exact reproduction requires the original local Yahoo JSON snapshots under `data/raw/`. These files are excluded from Git. [The acquisition record](docs/data_verification.json) contains retrieval metadata and checksums for NVDA, SPY and XLK: 2,766 observations each, retrieved on 11 October 2026 for 2015–2025.

A new clone alone cannot reproduce the recorded numerical results without those snapshots. A later vendor download may contain revised history and must be treated as a new dataset version. The acquisition script documents the original procedure, but currently writes a fixed audit path; do not rerun it against preserved evidence. A refresh requires separately versioned data and audit destinations. Data access does not confer redistribution rights; permitted submission and sharing arrangements remain to be established.

With the original snapshots available:

```sh
python scripts/check_snapshot.py
python -m pytest -q
jupyter lab
```

Run [notebook 01](notebooks/01_data_understanding.ipynb) for training-period exploration and [notebook 02](notebooks/02_dataset_and_naive_baseline.ipynb) for development windows and validation metrics. Keep executed copies in the ignored `artifacts/` directory. Generated notebook outputs are excluded from version control.

The calendar and baseline scripts write fixed, version-labelled destinations and refuse to overwrite them. Notebook 02 recomputes baseline results in memory. See the [benchmark reproduction notes](docs/MILESTONE_2.md) before running artifact-producing scripts.

## Repository structure
| Path | Responsibility |
|---|---|
| `configs/` | Forecast configuration and temporal boundaries |
| `src/market_forecasting/` | Reusable loading, validation, calendar and dataset logic |
| `scripts/` | Acquisition, integrity checks and experiment artifact generation |
| `notebooks/` | Explanatory analyses and technical demonstrations |
| `tests/` | Automated data-integrity, alignment and evaluation checks |
| `docs/` | Roadmap, progress, decisions, conventions and technical evidence |
| `data/` | Local vendor data, excluded from Git |
| `artifacts/` | Generated outputs, excluded from Git |

## Supporting documentation
- [Roadmap and completion criteria](docs/PLAN.md)
- [Progress and verification](docs/PROGRESS.md)
- [Research and engineering decisions](docs/DECISIONS.md)
- [Data conventions and limitations](docs/DATA_CONVENTIONS.md)

A Streamlit/Plotly demonstration, final report and assessment defence are planned after the forecasting experiments. The demonstration will present saved results; live trading and profitability claims are outside scope.
