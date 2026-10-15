# Development roadmap

This roadmap defines eight development milestones. Current status and supporting evidence belong in [PROGRESS.md](PROGRESS.md). Implementation and verification are tracked separately.

| Milestone | Objective and expected deliverables | Implementation and verification completion criteria |
|---|---|---|
| 1. Repository foundation | Repository structure, scope, configuration and data provenance | Structure and scope documented; configuration and provenance recorded; snapshot identity verifiable |
| 2. Environment and data audit | Reproducible dependencies, training exploration, calendar and corporate-action audit | Environment and exploration verified; session coverage checked; target defined; volume semantics resolved or an explicit scope decision approved |
| 3. Dataset construction | Target-date partitions, sequence construction and training-only transformations | Alignment and leakage checks pass; model-required scaling implemented and verified; development/test boundaries documented |
| 4. Baselines | Naive and ARIMA statistical forecasts | Forecast-origin protocol defined; both baselines executed on comparable validation dates; configurations, errors and limitations recorded |
| 5. Stock-only LSTM | Small initial LSTM and interpretable training workflow | Training, early stopping and training-only scaling verified; validation predictions and reproducible run metadata saved |
| 6. Controlled feature experiments | SMA, EMA, RSI and MACD candidates; controlled price-only, technical-indicator and SPY/XLK multivariate LSTM comparisons; bounded tuning and repeated seeds | Approved feature definitions and search protocol implemented; controlled comparisons and variability documented |
| 7. Final evaluation | Frozen approach, reserved 2025 evaluation, error analysis | Model selection frozen before test scoring; final results and limitations recorded without test-driven retuning |
| 8. Presentation and delivery | Dashboard, report, reproducibility check and defence preparation | Demonstration uses verified results; report and submission checked against requirements; reproduction and live-adjustment practice completed |

Technical summaries follow the roadmap numbering: [Milestone 2 — environment and audit](MILESTONE_2.md), [Milestone 3 — dataset construction](MILESTONE_3.md), [Milestone 4 — benchmarks](MILESTONE_4.md), and [Milestone 5 — LSTM](MILESTONE_5.md). Each document distinguishes implemented work from outstanding tasks; a summary does not imply milestone completion. The [progress tracker](PROGRESS.md) is the authoritative status record.

## Scope and dependencies
One forecast instrument (NVDA), daily observations and a next-session horizon. SPY and XLK are context series. No live trading, profit claims, news pipeline or LangChain component is planned. A second stock remains optional after the core study; configuration alone does not establish portability across exchanges or currencies.

Dataset validation precedes model fitting. Baselines precede LSTM comparisons. Approved feature and tuning protocols precede controlled experiments. Final testing follows model-selection freeze. Methodological alternatives and their rationale are recorded in [DECISIONS.md](DECISIONS.md).

## Assessment context
Selected topic: 1h, Time Series Forecasting with LSTM and Feature Engineering; topic submission has been confirmed. The recorded rubric allocation is technical implementation 15, innovation 8, performance 10, report 12, teamwork 5 and in-person Q&A/demonstration 50. Individual treatment of teamwork marks and the interpretation of forecasting-performance expectations remain clarification items.

## Milestone 6 experimental scope
Technical feature engineering is a core component. Compare price-only LSTM, LSTM with SMA/EMA/RSI/MACD features, and a multivariate variant adding SPY/XLK context. ARIMA remains a separate benchmark. Indicator parameters, causal calculations, warm-up and controlled comparison methods are specified in the [feature experiment proposal](FEATURE_EXPERIMENT_PROPOSAL.md), awaiting approval before implementation.

## Milestone 5 experiment scope
The [stock-only LSTM design](MILESTONE_5_DESIGN.md) specifies four controlled configurations: reference, increased units, dropout and reduced learning rate. Batch size is fixed at 32. Three paired seeds produce 12 internal-selection runs followed by three full-development fits. Hyperparameter selection and early stopping use the 2022 internal holdout; 2023–2024 remain external comparison data and 2025 remains reserved. All 15 fits are complete; [Milestone 5](MILESTONE_5.md) records configuration C, 19 refit epochs and the observed limitations. Milestone 6 remains unimplemented.
