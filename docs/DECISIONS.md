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

**Rationale and implications:** no fitted scaler is necessary for this baseline. Retaining earlier context across a split boundary is consistent with forecast-time availability. Previous validation closes become available for subsequent one-step forecasts; this is not a single 502-session forecast. Thirty sessions remains provisional for subsequent validation experiments. Results and limitations are recorded in [the technical summary](MILESTONE_2.md).
