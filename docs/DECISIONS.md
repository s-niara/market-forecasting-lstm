# Research and engineering decisions

Current implementation and verification status is maintained in [PROGRESS.md](PROGRESS.md); dataset definitions are maintained in [DATA_CONVENTIONS.md](DATA_CONVENTIONS.md).

## 2026-10-11 — Research scope and architecture
**Decision:** use topic 1h, Time Series Forecasting with LSTM and Feature Engineering, with NVDA as the initial case study. Use notebooks for explanation, Python modules for reusable logic and a later dashboard for presentation.

**Rationale and implications:** separation of analysis and reusable implementation supports reproducibility and technical demonstration. Ticker configuration and retraining support future extension but do not establish compatibility with every instrument. TensorFlow/Keras is the proposed modelling framework, subject to environment verification.

## 2026-10-11 — Data preservation and evaluation boundaries
**Decision:** preserve frozen vendor snapshots and exclude vendor data, environments and generated model artifacts from Git. Reserve 2025 for final forecast evaluation. Allow administrative data-quality checks without using test performance for selection.

**Rationale and implications:** fixed inputs permit verification of experiments while temporal separation limits selection leakage. Raw-data sharing remains subject to permitted submission and distribution arrangements. Git was initially local; no remote publication was performed as part of foundation development.

## 2026-10-11 — Reproducibility and provenance
**Decision:** record package versions, snapshot identities, consequential decisions and verification evidence.

**Rationale and implications:** versioned dependencies and input checksums identify the computational environment and data used in each experiment. The verified foundation environment used Python 3.12. Historical verification is retained in [PROGRESS.md](PROGRESS.md).

## 2026-10-12 — Forecast price convention
**Decision:** use Yahoo split-adjusted Close; retain dividend-adjusted Close as reference-only.

**Rationale and alternatives:** this target represents price forecasting while avoiding the mechanical discontinuity of an unadjusted split. Dividend-adjusted Close would change the economic target. Reconstructing original quotes would require separate point-in-time corporate-action handling. The chosen series is retrospective; do not apply split adjustments again or claim a point-in-time trading simulation.

## 2026-10-12 — Calendar audit and volume deferral
**Decision:** audit NASDAQ session coverage for NVDA and use the NYSE regular-session calendar as a documented proxy for SPY/XLK. Preserve half-days and exclude synthetic holiday rows. Defer volume features until adjustment semantics are independently corroborated.

**Rationale and implications:** matching instrument dates alone cannot detect a jointly missing session. The calendar-library check found no missing or unexpected dates, but does not constitute exchange certification. Insufficient evidence about volume warrants leaving the raw field unchanged and proceeding with price-only development.

## 2026-10-12 — Development windows and naive protocol
**Decision:** implement unscaled 30-session Close windows, assign partitions by target date, and stop construction at the end of validation. Evaluate the naive baseline using the previous observed session's Close.

**Rationale and implications:** no fitted scaler is necessary for this baseline. Retaining earlier context across a split boundary is consistent with forecast-time availability. Previous validation closes become available for subsequent one-step forecasts; this is not a single 502-session forecast. Thirty sessions remains provisional for subsequent validation experiments. Dataset construction is recorded in [Milestone 3](MILESTONE_3.md); forecast results and limitations are recorded in [Milestone 4](MILESTONE_4.md).

## 2026-10-13 — ARIMA statistical benchmark
**Decision:** compare nine nonseasonal ARIMA(p,1,q) models, p and q in {0,1,2}, without drift. Select by converged training AIC, fit on 2015–2022 and update filtered state after each one-step validation forecast without refitting coefficients.

**Rationale and alternatives:** a bounded order search limits complexity and keeps order selection independent of validation errors. First differencing addresses price-level persistence as a benchmark assumption, not a demonstrated stationarity result. Fixed coefficients separate parameter estimation from observation updates. Wider searches, drift and periodic refitting remain alternatives rather than unrecorded tuning. The unchanged price target and temporal boundaries permit direct comparison with the naive baseline. [Executed results](MILESTONE_4.md) show no improvement over naive forecasting.

## 2026-10-13 — Technical-feature experimental scope
**Decision:** retain SMA, EMA, RSI and MACD as candidate features and compare price-only, technical-indicator and multivariate SPY/XLK LSTMs. ARIMA is a comparator rather than a replacement for feature engineering.

**Rationale and implications:** nested input sets isolate the incremental contribution of technical features and market context. Shared eligibility, model settings and paired seeds reduce confounding. Detailed parameters and calculation conventions remain [proposed](FEATURE_EXPERIMENT_PROPOSAL.md); no feature or LSTM implementation is completed.

## 2026-10-14 — Bounded LSTM optimisation investigation
**Decision:** retain four configurations with paired seeds 42, 123 and 2026. Configuration D changes Adam learning rate from 0.001 to 0.0003 relative to A; all configurations use batch size 32. B varies hidden units and C varies post-LSTM dropout. Twelve internal-selection runs and three selected-configuration full-development fits form the planned 15-run budget.

**Rationale and implications:** one-factor contrasts isolate capacity, regularisation and optimiser step size within a common training budget. Batch-size effects and hyperparameter interactions are not investigated. The chronological 2022 holdout, early stopping and predefined selection rule are retained in the [design](MILESTONE_5_DESIGN.md). Loss histories, convergence, best epochs and stopping epochs remain required evidence; no LSTM run is yet completed.

## 2026-10-14 — Residual dollar-correction LSTM
**Decision:** predict a standardised next-session dollar correction and reconstruct price by adding its inverse-transformed value to the latest observed Close. Apply this formulation consistently to the Milestone 5 configurations and later feature variants.

**Rationale and alternatives:** direct-price prediction requires the recurrent network to reproduce unfamiliar price levels. A fixed persistence connection carries the known level while the network estimates departures from it. This does not remove input distribution shift or changing dollar volatility, and may underperform naive forecasting. The decision precedes new LSTM validation results; no direct-versus-residual experiment is added. The four configurations, paired seeds, 15-run budget, chronological protocol and selection rule remain unchanged.

## 14 October 2026 — Locked residual LSTM configuration
All four configurations completed three paired seeds. D had the lowest mean internal MAE, but all four were within the predefined 1% band. The rule first preferred 16 units, then selected C for the lowest across-seed MAE standard deviation. The selected dropout is 0.2, Adam learning rate 0.001 and batch size 32. Median best epoch 19 defines the three fresh full-development fits. The versioned [configuration](../configs/lstm_selected_v1.json) was written before external evaluation.

The mean external MAE of 1.705023 USD is worse than the naive 1.689494 USD. This does not authorise additional search; all 15 fits and their outcomes are retained. The limited seed count does not establish statistical superiority or globally optimal hyperparameters. Milestone 6 will retain this residual formulation, architecture and fixed training protocol, with separate weights and a new price-only control on the common feature-eligible interval. See [results and limitations](MILESTONE_5.md).
