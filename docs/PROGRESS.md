# Project progress

Last updated: 14 October 2026. Milestone numbers follow [PLAN.md](PLAN.md). Implementation and verification are tracked separately.

## Status overview
| Milestone | Implementation | Verification |
|---|---|---|
| 1. Repository foundation | Established | Foundation checks recorded |
| 2. Environment and data audit | Partial; volume unresolved | Environment, snapshots, calendar and notebook checked |
| 3. Dataset construction | Complete for price-only inputs | Scaling, alignment and leakage tests passed |
| 4. Baselines | Naive and ARIMA implemented | 17 tests pass; aligned validation benchmark executed |
| 5. Stock-only LSTM | Complete; 15/15 fits | 26 tests; notebook and artifacts verified |
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

**Repository status:** Foundation implementation is recorded in the initial repository commit. Subsequent experiment changes are maintained separately.

## 2. Environment and data audit

- [x] Verify the Python 3.12 environment and snapshot identities.
- [x] Explore training data and audit exchange-session coverage.
- [x] Define the split-adjusted Close target and document limitations.
- [ ] Corroborate volume adjustment semantics before using volume features.

**Objective/deliverables:** reproducible environment, training exploration, session coverage and explicit corporate-action conventions.

**Completed:** Python 3.12 environment recorded; three snapshots validated; training notebook executed; split-adjusted Close selected; independent calendar-library comparison performed.

**Files:** `requirements-foundation-lock.txt`, `requirements-milestone2-lock.txt`, `src/market_forecasting/data.py`, `src/market_forecasting/calendar.py`, `scripts/check_snapshot.py`, `scripts/audit_calendar.py`, notebook 01, both audit JSON files, `docs/DATA_CONVENTIONS.md` and [Milestone 2](MILESTONE_2.md).

**Verification/results:** each series has 2,766 full-period rows and 2,014 training sessions. Calendar coverage has no missing or unexpected dates. SPY/XLK use a documented NYSE proxy. Snapshot hashes match the preserved acquisition record in the latest implementation verification.

**Outstanding/next:** corroborate volume semantics before adding volume features; retain retrospective-adjustment and calendar-proxy limitations.

## 3. Dataset construction

- [x] Implement price-only windows and target-date partitions.
- [x] Test alignment, boundary context and exclusion of post-validation prices.
- [x] Implement and verify model-specific training-only scaling.

**Objective/deliverables:** chronological target-date partitions, forecasting windows, training-only transformations and leakage checks.

**Completed:** unscaled, single-feature, 30-session development windows; next-session targets; partition metadata; truncation before post-validation numerical processing.

**Files:** `src/market_forecasting/dataset.py`, `configs/nvda.json`, notebook 02, `tests/test_milestone2.py`; local window arrays and metadata in `artifacts/naive_v1/`. Technical summary: [Milestone 3](MILESTONE_3.md).

**Verification/results:** 1,984 training and 502 validation examples. Alignment, boundary context and invariance to post-validation price changes are tested. Calendar completeness is a separately audited precondition, not a builder guarantee.

**Completed extension:** Milestone 5 adds residual-target scaling in `preprocessing.py`, with `tests/test_lstm.py`. **Outstanding/next:** feature warm-up and common eligibility in Milestone 6. No final-test windows are produced by this development path.

## 4. Baselines

- [x] Implement and evaluate the naive validation baseline.
- [x] Verify error metrics against known values and save run metadata.
- [x] Define and implement training-AIC ARIMA selection with fixed-parameter state updates.
- [x] Verify comparable validation dates and forecast availability for both baselines.

**Objective/deliverables:** naive and traditional statistical forecasts under an explicit, comparable evaluation protocol.

**Completed:** naive and ARIMA forecasts, training-only ARIMA selection, rolling one-step validation, MAE/RMSE comparison, run metadata and explanatory notebooks.

**Files:** `scripts/build_naive_baseline.py`, `src/market_forecasting/dataset.py`, notebook 02, `tests/test_milestone2.py`, [Milestone 4](MILESTONE_4.md); ignored predictions and summary in `artifacts/naive_v1/`.

**Verification/results:** 502 validation forecasts; MAE **1.689494114 USD**, RMSE **2.589323812 USD** on the retrospective split-adjusted basis. These match the saved summary. Metric tests use known expected values. No test-period score is available.

**ARIMA result:** selected (2,1,2); validation MAE 1.720200573 USD and RMSE 2.612846480 USD, approximately 1.82% worse MAE than naive. All nine candidates converged; starting-parameter warnings are retained. See [Milestone 4](MILESTONE_4.md) and [run metadata](arima_validation_v1.json).

**Added files:** `src/market_forecasting/arima.py`, `scripts/run_arima_benchmark.py`, `configs/arima.json`, `tests/test_arima.py`, notebook 03 and `requirements-arima-lock.txt`.

**Next:** compare the completed residual LSTM in [Milestone 5](MILESTONE_5.md), then the approved future feature experiments.

## 5. Stock-only LSTM

- [x] Freeze four configurations, three paired seeds and residual formulation before training.
- [x] Implement training-only scaling, reconstruction, early stopping and experiment recording.
- [x] Complete 12 internal-selection fits and three fixed-epoch full-development fits.
- [x] Verify alignment, leakage safeguards, checkpoint restoration and reload equivalence.
- [x] Record all runs, loss curves, benchmark comparisons and limitations.
- [x] Execute the demonstration notebook without retraining.

**Objective/deliverables:** reproducible residual LSTM, controlled optimisation comparisons, locked configuration and external-validation results.

**Files:** `preprocessing.py`, `lstm.py`, experiment/report scripts, `tests/test_lstm.py`, search/selected configurations, LSTM environment lockfile, notebook 04, [design](MILESTONE_5_DESIGN.md), [technical summary](MILESTONE_5.md) and [results record](lstm_results_v1.json). Detailed histories, checkpoints and plots are local ignored artifacts.

**Results:** selected C (16 units, dropout 0.2, learning rate 0.001, batch 32), with 19 refit epochs. External MAE 1.705023 ± 0.013335 USD and RMSE 2.600560 ± 0.013501 USD across three seeds. Every seed is worse than naive on both metrics. All 15 fits completed; none failed or were replaced.

**Verification:** 26 tests pass; all 15 model reloads reproduce predictions exactly in the recorded environment. The 502 external target dates match the benchmarks. No 2025 forecasts were generated.

**Limitations/next:** single internal holdout, three seeds, narrow search and changing input/volatility distributions. No definitive causal explanation for poor performance. Feature definitions and common eligibility require approval before Milestone 6; no further fitting is included in this milestone.

## 6. Controlled feature experiments

- [x] Document SMA, EMA, RSI and MACD candidates and three controlled LSTM variants.
- [ ] Finalise the [proposed feature definitions and experimental protocol](FEATURE_EXPERIMENT_PROPOSAL.md) before implementation.
- [ ] Implement causal technical and market-context features.
- [ ] Run controlled comparisons, rolling validation and repeated seeds.
- [ ] Verify results and document variability and limitations.

**Objective/deliverables:** technical-feature and market-context variants, bounded tuning, rolling validation and repeated-seed comparisons.

**Completed/files/results:** [feature experiment proposal](FEATURE_EXPERIMENT_PROPOSAL.md) documents price-only, technical-indicator and multivariate SPY/XLK comparisons. Parameters, warm-up, causal formulas and evaluation methods await approval; no feature implementation or experiment result exists.

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
- First benchmark implementation run: 12 passing tests; snapshot checks, dependency consistency and notebook 02 execution recorded. See [dataset evidence](MILESTONE_3.md) and [benchmark evidence](MILESTONE_4.md).
- Documentation audit, 12 October 2026: compared stored baseline summary and calendar evidence with prose; both saved executed notebooks contained no error outputs. No new experiment was run during that audit.
- Prior documentation verification, 12 October 2026: 12 tests passed, all snapshot checksums and row counts verified, and documented baseline metrics matched saved evidence. Notebook code and outputs were preserved.
- Research-documentation verification, 12 October 2026: 12 automated tests passed; all three snapshot checksums and row counts verified; 43 local links resolved. In-memory recomputation reproduced the saved 502-forecast validation metrics exactly. All 38 files outside the cleanup scope were byte-identical, and both notebooks retained their code, outputs and metadata. No 2025 forecast evaluation was performed.
- Final documentation verification, 12 October 2026: 12 automated tests passed; all snapshot checksums and row counts verified; 43 local links resolved. Documented metrics matched the saved baseline summary. All 38 files outside the documentation scope were byte-identical; notebook code, outputs and metadata were preserved. No model experiment or test-period evaluation was run.

- ARIMA implementation verification, 13 October 2026: 17 tests passed. The development-only benchmark generated 502 forecasts on exactly the naive baseline target dates. Notebook 03 executed without errors, installed dependencies passed consistency checks, and all snapshot checksums matched. Source hashes and 53 local Markdown links were verified. Existing target configuration, dataset code, notebooks 01–02 and frozen audits were unchanged from the initial commit. Test-period performance remains uncomputed.

- Source-documentation verification, 14 October 2026: 17 tests passed and all three snapshot checksums/row counts matched. Executable syntax was equivalent before and after removing docstrings across all 13 Python files; 12 received documentation edits. All 21 protected data, configuration, notebook and artifact files were byte-identical. Ninety local Markdown links resolved. Historical ARIMA source hashes remain attached to the original run and were not rewritten. No LSTM training was performed.
- Documentation-style verification, 14 October 2026: simplified docstrings and moved essential contracts beside the relevant operations. All 17 tests passed; snapshot checksums and row counts matched. Executable syntax remained equivalent across 13 Python files after excluding docstrings, and 21 protected data/configuration/notebook/artifact files were unchanged. No experiment was rerun.

- Milestone 5 verification, 14 October 2026: 26 tests passed (five Keras/NumPy serialization deprecation warnings). All three snapshot hashes and 2,766-row counts matched; installed dependencies passed consistency checks. Fifteen complete run records and exact checkpoint-reload predictions were verified, with source/search/forecast configuration hashes unchanged. Notebook 04 executed all five code cells without errors. All 115 local documentation links resolved before this verification entry. No extra fitting or 2025 evaluation occurred.
