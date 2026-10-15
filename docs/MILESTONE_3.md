# Milestone 3: forecasting dataset construction

**Status: complete for the price-only representation.** Development windows, residual-target standardisation and associated alignment/leakage tests are implemented. Scaling was completed with [Milestone 5](MILESTONE_5.md). See [PLAN.md](PLAN.md) for completion criteria and [PROGRESS.md](PROGRESS.md) for current task status.

## Implemented representation
`src/market_forecasting/dataset.py` constructs 30-session, single-feature windows from split-adjusted Close, with the next session's Close as the target. The development dataset contains 2,486 examples with input shape `(2486, 30, 1)`: 1,984 training examples and 502 validation examples.

| Boundary | Recorded value |
|---|---|
| First development target | 2015-02-17 |
| First validation forecast origin | 2022-12-30 |
| First validation target | 2023-01-03 |
| Final development target | 2024-12-31 |

Thirty-session inputs require 30 preceding observations, leaving 1,984 training targets from 2,014 training observations. Each example records its window start, forecast origin, target date and partition. The initial 30-session length is provisional rather than a validated optimum.

## Temporal safeguards
Assign examples by target date. The final input belongs to the preceding forecast origin; historical context may cross a partition boundary because it is already available at forecast time. Previously observed validation closes may enter subsequent one-step inputs without revealing future targets.

The builder truncates to the end of validation before numerical checks and window construction. It does not construct 2025 forecast targets. Session completeness is an independently audited precondition: the builder checks ordering and numerical validity but does not enforce exchange-calendar coverage itself. See [Milestone 2](MILESTONE_2.md) and [data conventions](DATA_CONVENTIONS.md).

No fitted scaler is required for the naive baseline. Model-specific feature and target scaling must be fitted on training observations only. Descriptive chart rebasing in notebook 01 does not satisfy this modelling requirement.

## Implementation, evidence and reproduction
Implementation is in `src/market_forecasting/dataset.py`, configured by `configs/nvda.json`. [Notebook 02](../notebooks/02_dataset_and_naive_baseline.ipynb) demonstrates the representation and its use by the naive baseline. `tests/test_milestone2.py` retains its original filename and includes window-alignment, boundary-context and post-validation exclusion tests.

Local artifacts under `artifacts/naive_v1/` contain the development arrays, window metadata and a summary with configuration and snapshot hashes. These artifacts were generated together with the naive benchmark. Follow [README setup instructions](../README.md#installation) and run notebook 02 to reconstruct windows in memory. Artifact-writing behaviour is described in [Milestone 4](MILESTONE_4.md#evidence-and-reproduction).

Historical verification of the combined audit/dataset/naive implementation reported 12 passing tests, matching snapshot checksums, successful notebook 02 execution and consistent dependencies. Alignment and boundary tests establish the intended sample construction; a future-value perturbation test checks that post-validation prices cannot alter development windows. These checks cover specified failure modes rather than exhaustive correctness. Current verification is recorded in [PROGRESS.md](PROGRESS.md).

## Outstanding work
Training-only input and correction scaling is implemented in `src/market_forecasting/preprocessing.py` and verified in `tests/test_lstm.py`. Technical-feature warm-up and common eligibility rules remain [proposed for Milestone 6](FEATURE_EXPERIMENT_PROPOSAL.md), not part of the current price-only dataset. The existing development path produces no final-test windows.

Forecast metrics and model comparisons belong in [Milestone 4](MILESTONE_4.md).
