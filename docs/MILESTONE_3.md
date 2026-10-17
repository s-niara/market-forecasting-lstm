# Milestone 3: forecasting dataset construction

**Status: complete for the price-only representation.** Development windows, target-date partitions and alignment/leakage checks are implemented. Model-specific transformations are documented separately in [data conventions](DATA_CONVENTIONS.md). See [PLAN.md](PLAN.md) for completion criteria and [PROGRESS.md](PROGRESS.md) for implementation and verification status.

## Implemented representation
`src/market_forecasting/dataset.py` constructs 30-session, single-feature windows from split-adjusted Close, with the next session's Close as the target. The development dataset contains 2,486 examples with input shape `(2486, 30, 1)`: 1,984 training examples and 502 validation examples. The three axes represent examples, ordered historical sessions and input features.

| Boundary | Recorded value |
|---|---|
| First development target | 2015-02-17 |
| First validation forecast origin | 2022-12-30 |
| First validation target | 2023-01-03 |
| Final development target | 2024-12-31 |

Thirty-session inputs require 30 preceding observations, leaving 1,984 training targets from 2,014 training observations. The first 30 observations supply context; they are not erroneous records or deleted source data. Each example records its window start, forecast origin, target date and partition. Thirty sessions defines the common representation; it is not established as an optimal lookback.

## Loading and temporal safeguards
Notebook 02 verifies snapshot identity, loads NVDA through the reusable Yahoo loader with explicit development bounds, and independently audits NASDAQ session coverage. Price values are decoded only through 2024-12-31. The builder also truncates to the validation boundary before numerical checks and window construction. These separate controls restrict input access and enforce the development dataset contract; no 2025 forecast targets are constructed.

The builder requires a pandas Series with a chronological DatetimeIndex. A DataFrame is rejected rather than interpreted as a single price series. Ordering, uniqueness and numerical checks do not independently establish exchange-session completeness, which remains an audited precondition. See [Milestone 2](MILESTONE_2.md) and [data conventions](DATA_CONVENTIONS.md).

Assign examples by target date. The final input is the forecast-origin Close, and the label belongs to the next observed session. Historical context may cross a partition boundary because it is already available at forecast time. Earlier validation observations may enter subsequent inputs without revealing their future targets. Randomly splitting overlapping windows would not preserve this forecast-time separation.

Notebook assertions reconstruct each window's expected positions from the source index, compare all input values and labels, and check metadata against those positions. The first validation example demonstrates the 2022-12-30 origin and 2023-01-03 target across the New Year closure. These checks are based on every example's actual dates rather than only a single boundary case.

## Notebook demonstration
[Notebook 02](../notebooks/02_dataset_and_naive_baseline.ipynb) provides a twelve-section account of the research contract, bounded loading, session checks, chronological partitions, sliding windows, individual examples, boundary context, naive forecasts, rolling evaluation, metrics, error analysis and research implications. It imports the reusable implementation and traces its results against the historical observations. All executed tables and figures remain inline in the authoritative notebook.

No fitted scaler is required for the naive baseline. Model-specific feature and target transformations must use their permitted fitting observations only. Descriptive chart rebasing does not satisfy that modelling requirement. Training-only input and correction scaling is implemented in `preprocessing.py`; technical-feature warm-up and common eligibility are implemented in `features.py`. They are distinct from the unscaled price-only representation demonstrated here. The development path produces no final-test windows.

## Evidence and reproduction
Implementation is configured by `configs/nvda.json`. `tests/test_milestone2.py` retains its original filename and covers dataset alignment, boundary context, input contracts and exclusion of post-validation values. Follow the [README setup instructions](../README.md#installation) and execute Notebook 02 from beginning to end to reconstruct windows and naive validation predictions in memory.

The original artifacts under `artifacts/naive_v1/` contain development arrays, window metadata and a summary with configuration and snapshot hashes. They remain preserved. Recorded source hashes identify the original experimental implementation; input guards and notebook demonstrations do not rewrite that evidence. Running the notebook does not create another experiment directory, report or executed notebook copy. Artifact-producing scripts are described in [Milestone 4](MILESTONE_4.md#evidence-and-reproduction).

**Verification, 12 October 2026:** the combined audit/dataset/naive implementation reported 12 passing tests, matching snapshot checksums, successful Notebook 02 execution and consistent dependencies.

**Verification, 17 October 2026:** 103 automated tests passed across the data, dataset/baseline, ARIMA, LSTM and feature suites (five existing Keras/NumPy deprecation warnings), and all 23 Notebook 02 code cells executed in order with 5 inline figures and no errors. Bounded reconstruction preserves the `(2486, 30, 1)` inputs, target dates, partition counts and recorded naive MAE/RMSE. Per-window assertions verify all source-value and metadata alignments. These checks cover specified failure modes rather than exhaustive correctness; no model is fitted or test-period performance evaluated.

Forecast results and benchmark interpretation belong in [Milestone 4](MILESTONE_4.md).
