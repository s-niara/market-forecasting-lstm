# Development roadmap

This roadmap defines eight development milestones. Current status and supporting evidence belong in [PROGRESS.md](PROGRESS.md). Implementation and verification are tracked separately.

| Milestone | Objective and expected deliverables | Implementation and verification completion criteria |
|---|---|---|
| 1. Repository foundation | Repository structure, scope, configuration and data provenance | Structure and scope documented; configuration and provenance recorded; snapshot identity verifiable |
| 2. Environment and data audit | Reproducible dependencies, training exploration, calendar and corporate-action audit | Environment and exploration verified; session coverage checked; target defined; volume semantics resolved or an explicit scope decision approved |
| 3. Dataset construction | Target-date partitions, sequence construction and training-only transformations | Alignment and leakage checks pass; model-required scaling implemented and verified; development/test boundaries documented |
| 4. Baselines | Naive and traditional statistical forecasts | Forecast-origin protocol defined; both baselines executed on comparable validation dates; configurations, errors and limitations recorded |
| 5. Stock-only LSTM | Small initial LSTM and interpretable training workflow | Training, early stopping and training-only scaling verified; validation predictions and reproducible run metadata saved |
| 6. Controlled feature experiments | Technical and market-context variants, bounded tuning, rolling validation and repeated seeds | Approved feature definitions and search protocol implemented; controlled comparisons and variability documented |
| 7. Final evaluation | Frozen approach, reserved 2025 evaluation, error analysis | Model selection frozen before test scoring; final results and limitations recorded without test-driven retuning |
| 8. Presentation and delivery | Dashboard, report, reproducibility check and defence preparation | Demonstration uses verified results; report and submission checked against requirements; reproduction and live-adjustment practice completed |

The existing [Milestone 2 technical summary](MILESTONE_2.md) names an implementation increment spanning parts of roadmap milestones 2–4. Its existence does not mean those three roadmap milestones are complete. The progress tracker uses the eight roadmap numbers consistently.

## Scope and dependencies
One forecast instrument (NVDA), daily observations and a next-session horizon. SPY and XLK are context series. No live trading, profit claims, news pipeline or LangChain component is planned. A second stock remains optional after the core study; configuration alone does not establish portability across exchanges or currencies.

Dataset validation precedes model fitting. Baselines precede LSTM comparisons. Approved feature and tuning protocols precede controlled experiments. Final testing follows model-selection freeze. Methodological alternatives and their rationale are recorded in [DECISIONS.md](DECISIONS.md).

## Assessment context
Selected topic: 1h, Time Series Forecasting with LSTM and Feature Engineering; topic submission has been confirmed. The recorded rubric allocation is technical implementation 15, innovation 8, performance 10, report 12, teamwork 5 and in-person Q&A/demonstration 50. Individual treatment of teamwork marks and the interpretation of forecasting-performance expectations remain clarification items.
