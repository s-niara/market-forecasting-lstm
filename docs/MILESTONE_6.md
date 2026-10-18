# Milestone 6: controlled technical-feature experiments

**Experiments completed 15 October 2026; experiment verification completed 16 October 2026; notebook and implementation verification completed 18 October 2026.** Nine approved fits completed: three variants with seeds 42, 123 and 2026. All runs and outcomes are retained. No additional hyperparameter tuning or replacement runs were performed. No 2025 features, forecasts or performance metrics were calculated.

## Research comparison
- **A:** NVDA Close only, retrained on the common feature-eligible interval.
- **B:** A plus SMA(20,50) and EMA(12,26) price distances, RSI(14)/100, and Close-normalised MACD and histogram from MACD(12,26,9).
- **C:** B plus SPY and XLK one-session log Close returns.

The [approved feature specification](FEATURE_EXPERIMENT_PROPOSAL.md) defines causal formulas, arithmetic EMA seeds, Wilder RSI edge cases and minimum warm-up. The MACD signal is retained for visualisation and omitted from model inputs because signal = MACD − histogram. Volume remains excluded. Every input is available at its forecast origin; indicator states continue through the training/validation boundary without resets or future backfilling.

All variants retain the Milestone 5 architecture and training protocol: one stateless 16-unit LSTM, dropout 0.2, linear scalar output, Adam learning rate 0.001, batch size 32, 30 sessions and 19 fixed epochs. The output is a standardised dollar correction; inverse transformation and addition of the observed origin Close reconstruct the USD forecast. Each variant and seed fits independent weights. No validation loss is supplied to training and no per-variant early stopping occurs.

## Common eligibility and scaling
The first 49 feature rows remain unavailable. The first complete feature row is **2015-03-16**; the first 30-session forecast target is **2015-04-28** (source observation 80). All variants use exactly **1,935 training targets**, ending 2022-12-30, and **502 validation targets**, 2023-01-03 through 2024-12-31. Window shape is `(2437, 30, F)` for F = 1, 8 or 10. All fits perform 1,159 optimiser updates (61 batches × 19 epochs).

Feature scaling uses 1,964 unique fitting input rows, 2015-03-16 through 2022-12-29. Each column has its own training-only mean and population standard deviation. Validation is transformed with these fixed parameters without clipping. The common correction scaler uses only training labels: mean 0.007265633236838249 USD, standard deviation 0.3823882336925378 USD. Original prices and origins remain unscaled for reconstruction and metrics. Full statistics are retained in the [results record](feature_results_v1.json).

The common warm-up removes 49 training examples relative to Milestone 5. Consequently, the original Milestone 5 price-only result is not the control for feature effects; the newly fitted Model A is. Earlier LSTM results and artifacts remain unchanged. Naive and ARIMA retain their original documented procedures and histories; ARIMA was fitted using the full training history, whereas LSTM samples use finite windows and the common feature warm-up.

## Full validation comparison
| Model | Features | Parameters | MAE mean ± SD (USD) | RMSE mean ± SD (USD) |
|---|---:|---:|---:|---:|
| naive | — | — | 1.689494114 | 2.589323812 |
| arima | — | — | 1.720200573 | 2.612846480 |
| A | 1 | 1169 | 1.702599972 ± 0.012538326 | 2.599713430 ± 0.011934839 |
| B | 8 | 1617 | 1.717303573 ± 0.010323655 | 2.609909353 ± 0.008436726 |
| C | 10 | 1745 | 1.715304901 ± 0.010871786 | 2.606177381 ± 0.008966335 |

Standard deviations are sample deviations across three runs, not confidence intervals. Aggregate errors are means of individual metrics; no prediction ensemble is constructed. All nine LSTM runs have higher full-period MAE and RMSE than naive. Model A has the lowest average LSTM error; its mean MAE is approximately 0.78% above naive. All three LSTM means are below ARIMA's full-period means, but individual runs do not consistently outperform ARIMA.

| Model | Seed | MAE (USD) | RMSE (USD) |
|---|---:|---:|---:|
| A | 42 | 1.692170704 | 2.589635505 |
| A | 123 | 1.716511295 | 2.612892844 |
| A | 2026 | 1.699117917 | 2.596611941 |
| B | 42 | 1.708603847 | 2.601668459 |
| B | 123 | 1.714595537 | 2.609530454 |
| B | 2026 | 1.728711334 | 2.618529144 |
| C | 42 | 1.716668297 | 2.607558815 |
| C | 123 | 1.703815724 | 2.596600501 |
| C | 2026 | 1.725430681 | 2.614372827 |

## Paired feature contrasts
The following differences are in USD; negative means the added feature group improved that seed's metric.

| Seed | B − A MAE | B − A RMSE | C − B MAE | C − B RMSE |
|---|---:|---:|---:|---:|
| 42 | 0.016433 | 0.012033 | 0.008064 | 0.005890 |
| 123 | -0.001916 | -0.003362 | -0.010780 | -0.012930 |
| 2026 | 0.029593 | 0.021917 | -0.003281 | -0.004156 |

Adding technical indicators increases average MAE by approximately 0.014704 USD (0.86% relative to A). It improves seed 123 slightly, but worsens seeds 42 and 2026. Adding SPY/XLK reduces average MAE relative to B by approximately 0.001999 USD; this contrast improves two seeds and worsens one. Model C still has higher mean MAE and RMSE than A. Three seeds and one external development interval provide limited evidence; these numerical differences do not establish statistical significance or a general benefit from market context.

These are feature-group comparisons. They do not identify the importance or causal contribution of any individual indicator. Input dimensionality increases trainable parameters despite fixed hidden units, so the comparison includes this unavoidable capacity difference. Paired seed identifiers do not make initial weights identical across different input shapes. Equal 30-row sequences also differ in effective historical context: 30 SMA50 rows span 79 underlying closes, while EMA/RSI states retain earlier history. This is permitted causal information, but the comparison cannot isolate representation from that additional historical context.

## Annual comparison
| Year | Model | MAE (USD) | RMSE (USD) |
|---|---|---:|---:|
| 2023 | naive | 0.730728 | 1.022887 |
| 2023 | arima | 0.736197 | 1.022374 |
| 2023 | A (seed mean) | 0.742231 | 1.035047 |
| 2023 | B (seed mean) | 0.759861 | 1.050696 |
| 2023 | C (seed mean) | 0.763186 | 1.054011 |
| 2024 | naive | 2.640651 | 3.509699 |
| 2024 | arima | 2.696395 | 3.544404 |
| 2024 | A (seed mean) | 2.655347 | 3.521443 |
| 2024 | B (seed mean) | 2.667147 | 3.531851 |
| 2024 | C (seed mean) | 2.659868 | 3.525371 |

There are 250 targets in 2023 and 252 in 2024. Annual seed variability and every seed's annual metrics are retained in the results JSON and notebook. A has the lowest mean LSTM errors in both years. C is worse than B in 2023 and slightly better in 2024. Dollar errors are substantially larger in 2024; that observation alone cannot distinguish volatility, input shift and modelling limitations.

## Feature analysis and training behaviour
The following figures are generated from saved evidence:

- [Price, moving averages, normalised distances, RSI and MACD](../artifacts/features_v1/figures/technical_indicators.png): training-period 2021–2022 illustration.
- [Feature relationships](../artifacts/features_v1/figures/feature_relationships.png): correlations on unique fitting input rows, not importance scores.
- [Market-context returns](../artifacts/features_v1/figures/market_context.png): second half of 2022, illustrating correlated observed price returns.
- [Benchmark comparison](../artifacts/features_v1/figures/benchmark_comparison.png): all 502 dates, mean errors and seed variability.
- [Training curves](../artifacts/features_v1/figures/training_curves.png): inference-mode training MSE over all 19 epochs for every run.
- [Aligned cumulative errors](../artifacts/features_v1/figures/aligned_error_comparison.png): each seed and each variant's mean cumulative absolute-error difference from naive.

Moving-average distances and RSI/MACD features are strongly correlated in the training data. On the 1,964 unique fitting rows, SMA20 and EMA12 distances correlate at 0.956586, and SPY/XLK log returns at 0.938721. Moreover, `MACD/Close = 1/(1 + ema12_distance) - 1/(1 + ema26_distance)`, so relative MACD is exactly determined by the EMA distances. These relationships demonstrate redundancy, not that a specific feature caused the observed error or must be removed. Indicator and feature JSON tables retain dates and values for a future interactive HTML artefact; no interactive application is implemented in this milestone.

Models B and C reduce training inference loss more than A, but this does not translate into better average validation error. This is consistent with a generalisation limitation; training-only curves cannot locate an external overfitting onset. The experiment intentionally does not monitor validation each epoch or choose a favourable checkpoint. Every run uses its epoch-19 weights, including non-monotonic training outcomes.

## Evidence, verification and reproducibility
Reusable calculations are in `src/market_forecasting/features.py`; `scripts/run_feature_experiments.py` enforces the shared protocol and records failures if any occur. The unchanged Milestone 5 LSTM module supplies the architecture, chronological training loop and inference. `configs/feature_experiment_v1.json` fixes feature groups, seeds and hyperparameters. The existing `requirements-lstm-lock.txt` remains the runtime lockfile; no dependency changes were required.

Local evidence in `artifacts/features_v1/` includes all nine summaries, histories, saved `.keras` models, validation forecasts, raw indicators, transformed features, eligibility statistics and provenance. Provenance includes source/configuration/snapshot hashes, package versions, CPU execution and deterministic settings. Every saved model reproduced its original predictions exactly in this environment. Same-seed full training was not repeated beyond the nine-run budget, and cross-platform bitwise reproducibility is not established.

[Notebook 05](../notebooks/05_feature_experiments.ipynb) develops the indicator mathematics, verifies real feature construction and scaling, displays each 30-session input matrix and model summary, and demonstrates complete inference from saved checkpoints. Its 15 sections contain 18 executed code cells and 13 inline figures. All nine saved models are checked without training, with paired and annual error analysis, training histories and methodological limitations. The compact [results JSON](feature_results_v1.json) is project-facing evidence; generated artifacts and original snapshots remain excluded from Git. Preserve them for exact reproduction. JSON prediction exports use finite decimal precision; report checks agree with in-memory metrics within 1e-8 USD.

For deliberate independent reproduction in a separate copy with the original snapshots and empty Milestone 6 output destinations:

```sh
python scripts/run_feature_experiments.py
python scripts/report_feature_experiments.py
```

The runner refuses to overwrite the experiment directory before any fitting. The report also protects its figure directory and summary. Do not delete original evidence to rerun; no additional fits are authorised by this documentation. For ordinary review, execute notebook 05 only.

The [16 October verification record](feature_verification_v1.json) preserves the original evidence: 74 unchanged protected files, ten matching training-source hashes, exact reload predictions, 35 passing tests and six executed notebook cells. Those dated records continue to identify the original experiment. Session comparisons normalise datetime resolution, and saved forecast dates are explicitly parsed before alignment.

**Verification on 18 October 2026:** the full suite passes 132 tests, including nine additional checks for column-wise scaling, feature ordering, shared channels and rejection of inconsistent fixed-specification metadata before data access or training. All 18 notebook code cells execute from a fresh kernel with 13 inline figures, no errors and no notebook warnings. Five existing Keras/NumPy serialization deprecation warnings remain in automated tests. The notebook checks hand-calculated indicator values and edge cases, first-valid positions, common eligibility, unique-row scaling, inverse reconstruction, session alignment, prefix continuity and future perturbations. All nine checkpoints reproduce their 502 forecasts; the maximum difference from the rounded saved JSON is 4.9999 × 10⁻¹¹ USD. Parameter counts, feature order, optimiser settings, fixed 19 epochs, final-epoch training inference losses, annual results and aggregate metrics are reproduced. No research training, new experiment record or reserved-period numerical loading is required.

The feature runner validates the fixed SMA/EMA/RSI/MACD parameters, optimiser, loss and residual formulation before loading data. It rejects configuration metadata that would disagree with the hard-coded approved calculations. The approved configuration passes unchanged; this safeguard does not alter feature values, scalers, architectures, predictions or recorded results. The original source hashes remain attached to the original experiment rather than being replaced with current file hashes.

## Limitations and next decision
The fixed architecture and 19-epoch protocol isolate the approved feature comparison but may not suit each input group equally. More epochs, alternative feature subsets, longer burn-in, different return targets and additional seeds were not investigated. Minimum warm-up does not remove all EMA/RSI seed dependence. All technical indicators derive from the same price history, while SPY/XLK contain NVDA and are correlated context rather than independent information. The residual formulation still faces input distribution and dollar-volatility shifts. Existing retrospective vendor-history limitations remain.

The 2023–2024 period is repeatedly examined development evidence, not an untouched final assessment. No final methodology is frozen by this report. Milestone 7 and all 2025 feature computation, forecasting and evaluation remain blocked pending explicit approval of the final methodology.

## Files created or updated
Created: `src/market_forecasting/features.py`, `scripts/run_feature_experiments.py`, `scripts/report_feature_experiments.py`, `tests/test_features.py`, `configs/feature_experiment_v1.json`, notebook 05, this summary and `docs/feature_results_v1.json`. Updated: `README.md`, `START_HERE.md`, `docs/FEATURE_EXPERIMENT_PROPOSAL.md`, `docs/PLAN.md`, `docs/PROGRESS.md`, `docs/DECISIONS.md` and `docs/DATA_CONVENTIONS.md`. Verification evidence is recorded separately in `docs/feature_verification_v1.json`. Existing source modules and Milestone 5 artifacts were preserved. No files were staged or committed by the implementation workflow.
