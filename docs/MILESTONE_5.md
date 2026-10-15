# Milestone 5: residual price-only LSTM

**Completed: 14 October 2026.** All 15 approved fits completed, with no failed or replacement runs. The LSTM did not improve on the naive benchmark. No additional search was performed and no 2025 forecasts were generated.

## Method and selected configuration
The [predefined design](MILESTONE_5_DESIGN.md) fixes the four configurations, chronological holdout, training-only transformations, early stopping and selection rule. The network predicts a standardised next-session dollar correction from 30 historical closes; inverse transformation and addition of the last observed Close reconstruct the USD forecast. Price error equals correction error. This avoids requiring the network to reproduce the full price level but does not remove input distribution shift or changing dollar volatility.

The implementation separates preprocessing, model training and experiment orchestration. Histories record minibatch training loss, end-of-epoch training inference MSE and internal-validation MSE. Inference-mode training and validation losses use identical weights with dropout disabled; minibatch loss uses evolving weights and, for C, active dropout.

## Internal selection: 2022
| Configuration | Mean internal MAE | Seed SD | Mean internal RMSE | Best / stopping epochs (42, 123, 2026) |
|---|---:|---:|---:|---|
| A | 0.604546245 | 0.000690257 | 0.771849252 | 8/18, 1/11, 1/11 |
| B | 0.604579049 | 0.000708216 | 0.770825351 | 15/25, 14/24, 15/25 |
| C | 0.604467018 | 0.000100777 | 0.771285772 | 11/21, 19/29, 22/32 |
| D | 0.604009585 | 0.001026358 | 0.770653707 | 27/37, 1/11, 1/11 |

Each run has 251 internal target dates; fitting uses 1,733 windows with targets through 2021. Internal naive MAE is 0.602617545 USD and RMSE 0.767339286 USD; all LSTM configurations are worse on average.

D has the lowest mean MAE. All four means fall within the predefined 1% band (upper bound 0.610049681 USD). After preferring 16 units, C wins by its lower seed standard deviation. This band is a selection rule, not a statistical equivalence test. The selected [versioned configuration](../configs/lstm_selected_v1.json) therefore uses **16 units, dropout 0.2, Adam learning rate 0.001, batch size 32 and 19 refit epochs**. The epoch count is the median of C's best epochs (11, 19, 22). There are 1,169 trainable parameters with one feature.

## Loss curves and optimisation
[All training and validation curves](../artifacts/lstm_v1/figures/selection_curves.png), [internal-validation detail](../artifacts/lstm_v1/figures/validation_curves.png), and [fixed-epoch refit curves](../artifacts/lstm_v1/figures/final_training_curves.png) show every seed. Green markers identify best checkpoints; red markers identify stopping epochs in the selection plots. Detailed validation panels use individual y-axis scales.

Training inference MSE decreases only slightly from approximately one. Internal-validation MSE is much larger, around 7.6–7.8 in the training correction scale. The naive internal RMSE (0.7673 USD) is itself much larger than the fitting correction standard deviation (0.2789 USD), so this gap cannot be attributed solely to overfitting. Several runs continue to reduce training loss while validation loss deteriorates or fluctuates after its minimum, consistent with mild overfitting or optimisation drift. The curves do not establish clear underfitting or a unique explanation for the limited forecasting skill.

Increasing units produces no useful mean-MAE improvement within this search. Lowering the learning rate delays the best epoch for seed 42 from 8 to 27, but both other D seeds peak at epoch 1; it is not uniformly more stable. C has the lowest across-seed internal MAE spread, but three seeds cannot establish a general dropout benefit. Early stopping restores the observed minimum internal MSE; its independent benefit on unseen data was not separately tested. All 12 selection runs stop before the 80-epoch cap.

## External comparison: 2023–2024
| Model / seed | MAE (USD) | RMSE (USD) |
|---|---:|---:|
| naive | 1.689494114 | 2.589323812 |
| arima | 1.720200573 | 2.612846480 |
| LSTM C / 42 | 1.697893380 | 2.591403144 |
| LSTM C / 123 | 1.720407090 | 2.616064763 |
| LSTM C / 2026 | 1.696768498 | 2.594211961 |

Across the three fresh C fits, mean MAE is **1.705022990 ± 0.013334889 USD** and mean RMSE is **2.600559956 ± 0.013500802 USD** (sample standard deviations). These are averages of run metrics, not an ensemble of predictions. Mean MAE is approximately 0.92% worse than naive. Mean errors are below ARIMA, but seed 123 is worse than ARIMA on both metrics; no statistical superiority is established.

All models use the same 502 target dates, 2023-01-03 through 2024-12-31. Refits use 1,984 training examples, fresh training-only scalers and exactly 19 epochs (1,178 optimiser steps each). No external loss is used for stopping. The final records' `reached_epoch_cap` means the locked 19-epoch refit budget, not the selection cap of 80. Year-specific results are retained in the [results JSON](lstm_results_v1.json); dollar errors are larger in 2024, but this alone does not identify the cause.

## Evidence, environment and reproduction
[Notebook 04](../notebooks/04_lstm_experiments.ipynb) reads saved evidence and displays curves without fitting a model. `docs/lstm_results_v1.json` retains all run summaries, yearly metrics, selection evidence, scaling statistics and provenance. Detailed per-run summaries, histories, predictions and `.keras` checkpoints are stored under the ignored `artifacts/lstm_v1/selection/` and `artifacts/lstm_v1/final/` directories. Preserve these alongside the original data snapshots for exact reproduction; a Git clone alone lacks them. The executed notebook is `artifacts/04_lstm_experiments.executed.ipynb`.

Use the [README environment instructions](../README.md#installation) and `requirements-lstm-lock.txt`. Recorded runtime: Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1, macOS ARM CPU, two intra-operation threads and one inter-operation thread. Seeds and deterministic operations are explicitly set. Model reload predictions agree exactly in this environment. Full same-seed training was not repeated beyond the approved budget; bitwise cross-platform reproducibility is not established.

For a deliberate independent reproduction in a separate copy with empty output destinations, the original sequence is:

```sh
python scripts/run_lstm_experiments.py selection
python scripts/run_lstm_experiments.py final
python scripts/report_lstm_experiments.py
```

Check the runner's CLI before reproduction. Existing experiment directories and selected-configuration destinations are protected against overwrite. Do not delete preserved evidence to rerun. These commands perform 15 fits and are not needed to execute notebook 04. No further training is part of the completed milestone.

## Verification and remaining limitations
The 26-test suite checks alignment, unique-row scaling, correction inversion/reconstruction, zero-correction equivalence, future-value perturbations, chronological batches, checkpoint restoration, deterministic initialisation, selection rules and overwrite protection. Tests use synthetic data or mocked training and add no experimental fits. Saved checkpoint reloads were also verified in all 15 actual runs. Keras emits a NumPy `__array__` copy-keyword deprecation warning during serialization tests; tests still pass. Source/configuration hashes and original snapshot identities are verified separately.

The narrow search, single internal holdout, reuse of internal validation for stopping/selection, three seeds and transfer of a median epoch count to a larger fit all limit conclusions. The residual formulation does not guarantee profitable trading, improved accuracy or robust extrapolation. The scientific result is an unsuccessful attempt to improve the naive benchmark under a controlled protocol.

Milestone 6 remains unimplemented. It will retain the residual formulation and selected architecture/training settings, train each feature variant independently and rerun a price-only control on the common warm-up interval after the feature protocol is approved. Final 2025 evaluation remains reserved for Milestone 7.
