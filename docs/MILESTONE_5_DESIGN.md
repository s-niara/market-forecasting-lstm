# Milestone 5: stock-only LSTM experiment design

**Status: executed on 14 October 2026.** This document preserves the predefined protocol. All 15 approved fits completed; configuration C was selected with 19 refit epochs. See [Milestone 5 results](MILESTONE_5.md) for observed outcomes and verification. Future-tense descriptions below record the design established before training.

## Architecture and controlled configurations
Each model receives a (30, 1) window of NVDA split-adjusted Close and uses one forward, stateless LSTM, an optional post-LSTM Dropout layer and a single linear Dense output. Windows are independent samples; hidden state is not carried across batches. The Dense output represents a standardised next-session dollar correction. Inverse-transform the correction and add the latest observed Close to reconstruct the next-session price in USD. This fixed skip connection adds no trainable parameters.

| ID | LSTM units | Post-LSTM dropout | Batch size | Adam learning rate | Comparison with A |
|---|---|---|---|---|---|
| A | 16 | 0.0 | 32 | 0.001 | Reference |
| B | 32 | 0.0 | 32 | 0.001 | Capacity |
| C | 16 | 0.2 | 32 | 0.001 | Regularisation |
| D | 16 | 0.0 | 32 | 0.0003 | Learning rate |

The 16-unit architecture has 1,169 trainable parameters with one input feature; 32 units has 4,385. This limited capacity is appropriate to investigate before adding architectural depth. Adding the known latest Close reduces the need to reproduce unfamiliar price levels through the bounded recurrent representation. Input distribution shift and changing dollar volatility remain limitations; improved performance is not guaranteed.

Batch size is fixed: this study does not test a batch-size effect. D replaces the previously proposed batch-size contrast. The lower learning rate tests a smaller optimiser step while all other settings match A. A slower configuration may need more epochs; conclusions apply to the common budget rather than unlimited optimisation. Interactions between units, dropout and learning rate are not explored.

Common settings: lookback 30, Adam, maximum 80 epochs, chronological batch order (`shuffle=False`), and mean squared error on standardised dollar-correction targets. No stacked, bidirectional or attention layers are included. CPU execution and TensorFlow/Keras compatibility must be verified before training.

## Chronological internal validation
- Fit weights and transformations using training examples with target dates through 2021-12-31.
- Use 2022-01-03–2022-12-30 for internal validation, early stopping and configuration selection.
- Retain all available initial price history from 2015-01-02. There are 1,763 observations through 2021, producing 1,733 thirty-session fitting examples, followed by 251 internal-validation targets.
- Partition by target date, never by a random split of overlapping windows. Historical input context can cross the boundary; each input window ends before its target.
- During selection, load only development data through 2022 into the modelling path. Do not use 2023–2024 for configuration or stopping decisions, and do not evaluate 2025.

This single internal holdout replaces the earlier provisional three-fold design. It limits computation and preserves a substantial fitting period, but offers less evidence across market regimes. The internal-validation observations are reused for stopping and selection, so their scores are development evidence rather than an unbiased final assessment.

## Scaling and preprocessing
Use separate standardisation transforms for inputs and targets. Fit the input transform on unique observations occurring in fitting windows, rather than flattening overlapping windows and overweighting repeated values. Define each dollar-correction target as next-session Close minus the final Close in its input window. Fit the correction transform on fitting targets only. Store means, standard deviations and fitting-date boundaries.

For selection, no 2022 values enter fitted scaler statistics. `fit()` estimates transformation parameters; `transform()` applies fixed parameters to later observations. After configuration selection, fit fresh transforms on the appropriate 2015–2022 fitting rows and targets. Apply these unchanged to 2023–2024. `inverse_transform()` returns the predicted correction to dollars; add the unscaled origin Close before evaluating price errors. A zero-dollar correction reproduces the naive forecast, whereas zero standardised output corresponds to the mean training correction. Price error equals correction error exactly; standardised correction MSE is dollar price MSE divided by the fixed fitting correction variance.

Do not clip validation values to the fitting range. Preserve the approved dollar-correction representation. The calendar audit remains a prerequisite; scaling does not establish session completeness or eliminate distribution shift.

## Early stopping and history recording
Monitor internal-validation standardised MSE with patience 10, minimum improvement 0, maximum 80 epochs and restoration of the lowest-validation-MSE checkpoint. Record the best epoch, stopping epoch, cap-reached status, optimiser steps and elapsed time for every run. Abort and record nonfinite training rather than replacing a failed seed with a favourable run.

Save ordinary training loss and internal-validation loss at every epoch. Also evaluate training MSE after each epoch in inference mode, with dropout disabled and the same weights used for validation evaluation. The latter provides a consistent objective comparison; ordinary training loss averages minibatches while weights change and dropout may be active.

Plot loss curves with best/stopping epochs marked. Compare convergence rate, variability and checkpoint-versus-terminal behaviour. Report overfitting or underfitting only when supported; neither a loss gap nor a poor forecast alone establishes its cause. Reaching the cap does not automatically authorise additional search or training runs.

## Repeated seeds and selection
Use paired seeds **42, 123 and 2026** for each of A–D: 12 selection runs. Record each run, mean and sample standard deviation of internal-validation USD MAE/RMSE. Do not select the best seed.

Predefined selection rule:
1. Require three valid, finite runs for an eligible configuration; retain failures in the experiment record.
2. Find the lowest mean internal-validation MAE across eligible configurations.
3. Treat means within 1% of that minimum as practically close, without interpreting the band as statistical equivalence.
4. Within that group, prefer fewer LSTM units, then lower MAE standard deviation, then lower mean MAE, then configuration ID.

Use the selected configuration's median best epoch as a fixed duration for three fresh full-development fits. Train each seed on all eligible 2015–2022 examples, including the former internal holdout, with newly fitted training-only scalers. Do not use external validation for early stopping. This yields **12 selection runs + 3 configured-model runs = 15 training runs**. Transferring an epoch count to a larger sample is a documented approximation, not an optimality guarantee.

Record seeds, deterministic-operation settings, package versions, execution hardware, data/code hashes, checkpoints, histories, predictions and scaler parameters. Deterministic reproducibility must be checked on the chosen runtime; identical results across different platforms are not guaranteed. Infrastructure tests must not become an additional undocumented model search.

## External evaluation and configuration lock
Only after selection, evaluate the three full-development fits on the same 502 target dates as the naive and ARIMA benchmarks: 2023-01-03–2024-12-31. Each one-step forecast uses the previous 30 observed closes. Prior external-validation observations may enter later input windows after becoming available; weights and scalers remain fixed.

Report inverse-transformed USD MAE/RMSE per seed, aggregate variability, aligned benchmark errors, and separate 2023/2024 summaries. Do not create an ensemble by silently averaging predictions. No profitability, statistical-significance or globally optimal architecture claim follows from these comparisons.

Write a versioned selected configuration only after the selection rule is executed. Include architecture, learning rate, batch size, dropout, lookback, seeds, objective, scaling definitions, epoch budget and selection evidence. A disappointing external-validation result does not automatically reopen this search.

## Milestone 6 handoff
Use this same residual reconstruction and keep the selected architecture and training hyperparameters fixed across the price-only, technical-indicator and SPY/XLK variants. Train separate weights and transforms per variant using the same fitting eligibility, temporal boundaries and seeds.

Indicator warm-up requires a new price-only control on the common complete-feature interval; preserve the original Milestone 5 run as a separate experiment. Input dimensionality changes parameter counts even with the same hidden units. Exact indicator methods and eligibility rules in the [feature proposal](FEATURE_EXPERIMENT_PROPOSAL.md) still require approval before feature implementation. No Milestone 6 training is included in the 15-run Milestone 5 budget.

## Planned implementation and verification
Reusable modules will separate preprocessing, architecture/training and experiment orchestration. A runner will distinguish internal selection from external evaluation, while notebook 04 will display histories and interpret recorded results. Planned artifacts include a search configuration, selected configuration, environment lock, local checkpoints/predictions and a Milestone 5 technical results summary.

Tests will cover fitting boundaries, unique-row scaling, inverse transforms, target-date alignment, future-value perturbations, checkpoint restoration, deterministic selection/ties, model reload equivalence and protection against overwriting prior runs. The existing benchmark tests remain required. Short purpose docstrings and concise comments beside the relevant operations will explain the methodology and its limits using the [project code-documentation standard](../README.md#code-documentation-standard).
