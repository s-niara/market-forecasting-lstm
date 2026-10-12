# Milestone 2 implementation increment: audit and first benchmark

This historical implementation increment covers parts of roadmap milestones **2 (audit), 3 (dataset) and 4 (baselines)**. It is not a completion claim for all three. See [PLAN.md](PLAN.md) and [PROGRESS.md](PROGRESS.md).

## Training-data exploration
Notebook 01 loads the configuration and frozen snapshots, converts timestamps to exchange-local dates, validates the data and selects training rows. Each series contains 2,014 training sessions from 2015-01-02 to 2022-12-30. NVDA's first split-adjusted Close is 0.50325 USD; this is not the original quoted price.

The comparative chart rebases each Close series to 100 on its first training date. This is a descriptive transformation, not model scaling or a dividend-reinvested total-return series. NVDA's growth motivates later consideration of distribution change; it does not demonstrate predictability. Target, calendar and volume definitions are specified in [DATA_CONVENTIONS.md](DATA_CONVENTIONS.md).

## Implemented dataset and benchmark
The builder produces 2,486 development examples with input shape `(2486, 30, 1)`: 1,984 training examples and 502 validation examples. The first target is 2015-02-17. The first validation origin is 2022-12-30 and its target is 2023-01-03. The final development target is 2024-12-31.

The naive forecast equals the last Close in each input window. Only that final value is required by the baseline; the shared window representation supports later model comparisons. The evaluation is rolling one-step forecasting, using prior observations as they become available.

| Validation result | Value |
|---|---|
| Forecast count | 502 |
| MAE, retrospective split-adjusted USD | 1.689494114 |
| RMSE, retrospective split-adjusted USD | 2.589323812 |

These values correspond to the saved local `artifacts/naive_v1/summary.json`, which records configuration and snapshot hashes. They are validation errors, not test scores, profit estimates or evidence that a learned model improves on the baseline. Dollar errors depend on the price scale. Comparisons must use consistent targets, dates and forecast availability. Close visual agreement is expected for a previous-close forecast and does not establish predictive skill.

## Verification and evidence
The latest implementation verification reported 12 passing automated tests, matching checksums for all three snapshots, successful notebook execution and consistent installed dependencies. Tests cover alignment, partition context, exclusion of post-validation price values, calendar exceptions and known metric values. Both saved executed notebooks contain no error outputs. These checks cover specified failure modes rather than establishing exhaustive correctness.

The [calendar audit](calendar_audit_2026-10-12.json) records date coverage independently of the price metrics. No 2025 forecasts have been evaluated. Full-snapshot administrative checks predate model construction and are documented separately in [data verification](data_verification.json).

## Reproduction
Follow [README installation and data prerequisites](../README.md#installation). With original snapshots, run notebook 02 to recompute development windows and validation metrics without overwriting stored artifacts.

`scripts/audit_calendar.py` writes `docs/calendar_audit_2026-10-12.json`; `scripts/build_naive_baseline.py` writes `artifacts/naive_v1/`. These fixed destinations refuse overwriting. They are not a general experiment-versioning interface. A deliberate new artifact run requires new destinations with a recorded rationale; preserve existing evidence. Executed notebook copies remain local under `artifacts/`.

## Methodological interpretation
Thirty-session inputs require 30 preceding observations, leaving 1,984 training targets from 2,014 training observations. Each prediction is indexed by its target date; the final input belongs to the preceding forecast origin. Pre-boundary context and earlier observed validation closes are therefore available without using future target values.

Chart rebasing is descriptive and does not fit a model transformation. Subsequent model scaling must use training observations only. Retrospective price adjustment, the calendar proxy and unresolved volume semantics remain limitations, as detailed in [data conventions](DATA_CONVENTIONS.md).

## Outstanding work
Volume verification, model-specific scaling, the traditional baseline, all LSTM experiments and final test evaluation remain outstanding. The central [progress tracker](PROGRESS.md) owns subsequent status updates.
