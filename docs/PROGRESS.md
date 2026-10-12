# Project progress

Last updated: 12 October 2026. Milestone numbers follow [PLAN.md](PLAN.md). Implementation and verification are tracked separately.

## Status overview
| Milestone | Implementation | Verification |
|---|---|---|
| 1. Repository foundation | Established | Foundation checks recorded |
| 2. Environment and data audit | Partial; volume unresolved | Environment, snapshots, calendar and notebook checked |
| 3. Dataset construction | Partial; unscaled price-only windows | Implemented window and leakage checks passed |
| 4. Baselines | Partial; naive only | Naive validation run and metric tests completed |
| 5. Stock-only LSTM | Not started | Not performed |
| 6. Controlled feature experiments | Not started | Not performed |
| 7. Final evaluation | Not started; reserved | No test forecast evaluation |
| 8. Presentation and delivery | Not started; supporting notes exist | Delivery verification outstanding |

## 1. Repository foundation

- [x] Establish package structure, scope and configuration.
- [x] Record snapshot provenance and reproducible environment files.
- [x] Verify foundation data-loading tests and notebook execution.

**Objective/deliverables:** research scope, repository structure, configuration and traceable data provenance.

**Completed:** local repository established; package structure, ignored data/artifact locations, configuration, acquisition checksums and documentation created. Research documentation is organised by purpose.

**Files:** `README.md`, `START_HERE.md`, `pyproject.toml`, `configs/nvda.json`, `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/data_verification.json`, foundation lockfile and package structure.

**Verification/results:** historical foundation verification recorded six passing tests, verified snapshots and successful notebook 01 execution. No model-performance result belongs to this milestone.

**Repository status:** The repository has been reinitialised and currently has zero commits; existing research files and historical experiment evidence predate the new Git history.

## 2. Environment and data audit

- [x] Verify the Python 3.12 environment and snapshot identities.
- [x] Explore training data and audit exchange-session coverage.
- [x] Define the split-adjusted Close target and document limitations.
- [ ] Corroborate volume adjustment semantics before using volume features.

**Objective/deliverables:** reproducible environment, training exploration, session coverage and explicit corporate-action conventions.

**Completed:** Python 3.12 environment recorded; three snapshots validated; training notebook executed; split-adjusted Close selected; independent calendar-library comparison performed.

**Files:** `requirements-foundation-lock.txt`, `requirements-milestone2-lock.txt`, `src/market_forecasting/data.py`, `src/market_forecasting/calendar.py`, `scripts/check_snapshot.py`, `scripts/audit_calendar.py`, notebook 01, both audit JSON files and `docs/DATA_CONVENTIONS.md`.

**Verification/results:** each series has 2,766 full-period rows and 2,014 training sessions. Calendar coverage has no missing or unexpected dates. SPY/XLK use a documented NYSE proxy. Snapshot hashes match the preserved acquisition record in the latest implementation verification.

**Outstanding/next:** corroborate volume semantics before adding volume features; retain retrospective-adjustment and calendar-proxy limitations.

## 3. Dataset construction

- [x] Implement price-only windows and target-date partitions.
- [x] Test alignment, boundary context and exclusion of post-validation prices.
- [ ] Implement and verify model-specific training-only scaling.

**Objective/deliverables:** chronological target-date partitions, forecasting windows, training-only transformations and leakage checks.

**Completed:** unscaled, single-feature, 30-session development windows; next-session targets; partition metadata; truncation before post-validation numerical processing.

**Files:** `src/market_forecasting/dataset.py`, `configs/nvda.json`, notebook 02, `tests/test_milestone2.py`; local window arrays and metadata in `artifacts/naive_v1/`.

**Verification/results:** 1,984 training and 502 validation examples. Alignment, boundary context and invariance to post-validation price changes are tested. Calendar completeness is a separately audited precondition, not a builder guarantee.

**Outstanding/next:** model-specific scaling and corresponding verification. No final-test windows are produced by this development path.

## 4. Baselines

- [x] Implement and evaluate the naive validation baseline.
- [x] Verify error metrics against known values and save run metadata.
- [ ] Define the ARIMA fitting/refitting protocol and implement the statistical baseline.
- [ ] Verify comparable validation dates and forecast availability for both baselines.

**Objective/deliverables:** naive and traditional statistical forecasts under an explicit, comparable evaluation protocol.

**Completed:** previous-session-Close forecast, rolling one-step validation, MAE/RMSE calculation, saved metadata and explanatory notebook.

**Files:** `scripts/build_naive_baseline.py`, `src/market_forecasting/dataset.py`, notebook 02, `tests/test_milestone2.py`, `docs/MILESTONE_2.md`; ignored predictions and summary in `artifacts/naive_v1/`.

**Verification/results:** 502 validation forecasts; MAE **1.689494114 USD**, RMSE **2.589323812 USD** on the retrospective split-adjusted basis. These match the saved summary. Metric tests use known expected values. No test-period score is available.

**Outstanding/next:** define and implement the ARIMA fitting/refitting protocol and comparable validation evaluation. ARIMA settings have not been selected.

## 5. Stock-only LSTM

- [ ] Verify modelling dependencies and specify the initial architecture.
- [ ] Implement scaling, LSTM fitting and early stopping.
- [ ] Test temporal safeguards and save validation predictions and training history.

**Objective/deliverables:** small stock-only LSTM, early stopping, training history and reproducible predictions.

**Completed/files/results:** no model implementation or executed result. Configuration and documentation provide preparatory context only.

**Outstanding/verification/next:** verify modelling dependencies, specify and justify model design choices, implement scaling and fitting, test temporal safeguards and evaluate on validation data.

## 6. Controlled feature experiments

- [ ] Specify feature definitions and a bounded experiment protocol.
- [ ] Implement causal technical and market-context features.
- [ ] Run controlled comparisons, rolling validation and repeated seeds.
- [ ] Verify results and document variability and limitations.

**Objective/deliverables:** technical-feature and market-context variants, bounded tuning, rolling validation and repeated-seed comparisons.

**Completed/files/results:** candidate inputs described in the roadmap; no implemented experiment or result.

**Outstanding/verification/next:** specify exact feature definitions and comparison protocol, resolve volume inclusion, implement causal transformations and run controlled experiments.

## 7. Final evaluation

- [ ] Freeze model selection and experiment settings.
- [ ] Evaluate the reserved 2025 period after the freeze.
- [ ] Verify final metrics and document error analysis and limitations.

**Objective/deliverables:** frozen configuration, 2025 forecast evaluation, error analysis and limitations.

**Completed/files/results:** test reservation is recorded in configuration and conventions; no final evaluation implementation or result.

**Outstanding/verification/next:** finish validation-based selection and record the freeze before accessing test forecast performance. Administrative snapshot checks are not test evaluation.

## 8. Presentation and delivery

- [ ] Build a demonstration from verified results and prepare the report.
- [ ] Confirm assessment and data-sharing requirements.
- [ ] Verify reproduction and practise technical demonstration and defence.

**Objective/deliverables:** results dashboard, final report, reproducibility check and defence practice.

**Completed/files/results:** notebooks and technical notes provide supporting material; no completed dashboard, final report or defence assessment.

**Outstanding/verification/next:** prepare delivery after experiments; confirm individual teamwork-mark treatment and assessment expectations; verify submission requirements, demonstration and reproduction.

## Verification record
- Foundation run: six passing tests; snapshot checks and notebook 01 execution recorded.
- First benchmark implementation run: 12 passing tests; snapshot checks, dependency consistency and notebook 02 execution recorded. See [technical evidence](MILESTONE_2.md).
- Documentation audit, 12 October 2026: compared stored baseline summary and calendar evidence with prose; both saved executed notebooks contained no error outputs. No new experiment was run during that audit.
- Prior documentation verification, 12 October 2026: 12 tests passed, all snapshot checksums and row counts verified, and documented baseline metrics matched saved evidence. Notebook code and outputs were preserved.
- Research-documentation verification, 12 October 2026: 12 automated tests passed; all three snapshot checksums and row counts verified; 43 local links resolved. In-memory recomputation reproduced the saved 502-forecast validation metrics exactly. All 38 files outside the cleanup scope were byte-identical, and both notebooks retained their code, outputs and metadata. No 2025 forecast evaluation was performed.
- Final documentation verification, 12 October 2026: 12 automated tests passed; all snapshot checksums and row counts verified; 43 local links resolved. Documented metrics matched the saved baseline summary. All 38 files outside the documentation scope were byte-identical; notebook code, outputs and metadata were preserved. No model experiment or test-period evaluation was run.
