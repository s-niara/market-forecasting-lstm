# Project progress

Last updated: 18 October 2026. Milestone numbers follow [PLAN.md](PLAN.md). Implementation and verification are tracked separately.

## Status overview
| Milestone | Implementation | Verification |
|---|---|---|
| 1. Repository foundation | Established | Foundation checks recorded |
| 2. Environment and data audit | Partial; volume unresolved | Environment, snapshots, calendar and notebook checked |
| 3. Dataset construction | Complete for price-only inputs | Bounded loading, window alignment, input contracts and leakage checks passed |
| 4. Baselines | Naive and ARIMA implemented | Recorded forecasts preserved; Notebook 02 metrics and alignment verified |
| 5. Stock-only LSTM | Complete; 15/15 fits | 123 tests; 15-checkpoint inference, scaling and executed notebook verified |
| 6. Controlled feature experiments | Complete; 9/9 fits | 132 tests; nine-checkpoint inference, feature arithmetic, common samples and executed notebook verified |
| 7. Final evaluation | Not started; reserved | No test forecast evaluation |
| 8. Presentation and delivery | Not started; supporting notes exist | Delivery verification outstanding |

## 1. Repository foundation

- [x] Establish package structure, scope and configuration.
- [x] Define notebooks as the primary research narrative and demonstrations, backed by reusable Python modules.
- [x] Record snapshot provenance and reproducible environment files.
- [x] Verify foundation data-loading tests and notebook execution.

**Objective/deliverables:** research scope, repository structure, configuration and traceable data provenance.

**Completed:** local repository established; package structure, ignored data/artifact locations, configuration, acquisition checksums and documentation created. Research documentation is organised by purpose. Notebooks execute the actual Python functions; important algorithms have one authoritative implementation, and analytical findings inform implementation decisions.

**Files:** `README.md`, `START_HERE.md`, `pyproject.toml`, `configs/nvda.json`, `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/data_verification.json`, foundation lockfile and package structure.

**Verification/results:** historical foundation verification recorded six passing tests, verified snapshots and successful notebook 01 execution. No model-performance result belongs to this milestone.

**Repository status:** Foundation implementation is recorded in the initial repository commit. Subsequent experiment changes are maintained separately.

## 2. Environment and data audit

- [x] Verify the Python 3.12 environment and snapshot identities.
- [x] Explore training data and audit exchange-session coverage.
- [x] Implement bounded exploratory loading and shared validation diagnostics.
- [x] Demonstrate validation rejection, calendar coverage and critical EDA interpretation inline.
- [x] Retain one authoritative Notebook 01 with executed outputs and six inline figures.
- [x] Define the split-adjusted Close target and document limitations.
- [ ] Corroborate volume adjustment semantics before using volume features.

**Objective/deliverables:** reproducible environment, training exploration, session coverage and explicit corporate-action conventions.

**Completed:** Python 3.12 environment recorded; frozen snapshots verified; split-adjusted Close defined; training values loaded through explicit temporal bounds and checked with shared financial diagnostics plus independent exchange-calendar comparisons. Notebook 01 documents acquisition and conventions, examines structure and data quality, and analyses price levels, dollar changes, percentage/log returns, distributions, rolling volatility and cross-instrument relationships. It demonstrates rejection of invalid records without altering the source data.

**Files:** `requirements-foundation-lock.txt`, `requirements-milestone2-lock.txt`, `src/market_forecasting/data.py`, `src/market_forecasting/calendar.py`, `scripts/check_snapshot.py`, `scripts/audit_calendar.py`, notebook 01, both audit JSON files, `docs/DATA_CONVENTIONS.md` and [Milestone 2](MILESTONE_2.md).

**Verification, 16 October 2026:** 58 relevant automated tests passed; all 14 Notebook 01 code cells executed in order in a fresh kernel with six inline figures and no errors. All source checksums matched. The 2,014 training observations per instrument, from 2015-01-02 to 2022-12-30, were reproduced exactly, including values, dtypes and exchange-local dates. Financial checks found no null/nonfinite cells, duplicate dates, non-positive prices, negative volumes or inconsistent OHLC rows. Calendar comparisons found no missing or unexpected training sessions; SPY/XLK use the documented NYSE proxy. Synthetic tests verify excluded quote tokens are not decoded, invalid selected records are rejected, and aligned instruments can still share a missing session. Tests and calculations are reproducible in the existing environment without separate EDA exports.

**EDA findings:** NVDA daily percentage-return standard deviation is 3.030 percentage points, with 126 returns beyond conventional IQR fences; these observations are retained. Daily dollar-change SD increases from 0.0129 USD in 2015 to 0.7666 USD in 2022, while return SD increases from 2.177 to 3.982 percentage points. NVDA log-return correlations are 0.655 with SPY and 0.745 with XLK. These descriptive findings identify scale changes and associations, not predictive improvement. Full-precision calculations and interpretation are retained in Notebook 01.

**Provenance:** the preserved acquisition and calendar records contain 2,766 full-period rows per instrument. Those historical administrative records are distinct from this training-only analysis; no reserved-period price or forecasting performance is inspected.

**Outstanding/next:** corroborate volume semantics before adding volume features; retain retrospective-adjustment and calendar-proxy limitations.

## 3. Dataset construction

- [x] Implement price-only windows and target-date partitions.
- [x] Test alignment, boundary context and exclusion of post-validation prices.
- [x] Use bounded development loading, snapshot provenance and calendar preflight in Notebook 02.
- [x] Trace every window against source observations and demonstrate the one-dimensional input contract.
- [x] Implement and verify model-specific training-only scaling.

**Objective/deliverables:** chronological target-date partitions, forecasting windows, training-only transformations and leakage checks.

**Completed:** unscaled, single-feature, 30-session development windows; next-session targets; partition metadata; bounded quote loading through the development endpoint and separate calendar verification. The builder rejects non-Series inputs and truncates before post-validation numerical processing. Notebook 02 explains the representation and verifies input values, target values and metadata dynamically for every window.

**Files:** `src/market_forecasting/dataset.py`, `configs/nvda.json`, notebook 02, `tests/test_milestone2.py`; local window arrays and metadata in `artifacts/naive_v1/`. Technical summary: [Milestone 3](MILESTONE_3.md).

**Verification/results:** 1,984 training and 502 validation examples. Alignment, boundary context and invariance to post-validation price changes are tested. Calendar completeness is a separately audited precondition, not a builder guarantee.

**Related implementation:** Milestone 5 supplies residual-target scaling in `preprocessing.py`, with `tests/test_lstm.py`; Milestone 6 supplies feature warm-up and common eligibility. No final-test windows are produced by this development path.

**Verification, 12 October 2026:** the combined dataset/baseline implementation passed 12 tests, dependency and snapshot checks, and Notebook 02 execution. In-memory recomputation reproduced the saved 502-forecast validation metrics exactly.

**Verification, 17 October 2026:** 103 automated tests passed across the data, dataset/baseline, ARIMA, LSTM and feature suites (five existing Keras/NumPy deprecation warnings). All 23 Notebook 02 code cells executed in order in a fresh kernel with 5 inline figures and no errors. Bounded loading and calendar checks preserve 2,486 windows of shape `(30, 1)`, including 1,984 training and 502 validation examples. Every window, target and metadata row is checked against source positions. The first validation forecast uses 2022-12-30 Close for the 2023-01-03 target. No reserved-period quote values are decoded or evaluated.

## 4. Baselines

- [x] Implement and evaluate the naive validation baseline.
- [x] Verify error metrics against known values and save run metadata.
- [x] Define and implement training-AIC ARIMA selection with fixed-parameter state updates.
- [x] Verify comparable validation dates and forecast availability for both baselines.
- [x] Present the naive protocol, metric calculations and validation error analysis in the authoritative executed Notebook 02.
- [x] Present training-period differencing, stationarity and correlation diagnostics in Notebook 03.
- [x] Reproduce the predetermined ARIMA order and verify coefficients, AIC and all recorded rolling forecasts.
- [x] Trace forecast-before-observation events, fixed coefficients and filtered-state updates.
- [x] Examine training innovations with explicit initialization exclusion and diagnostic limitations.

**Objective/deliverables:** naive and traditional statistical forecasts under an explicit, comparable evaluation protocol.

**Completed:** naive and ARIMA forecasts, training-only ARIMA selection, rolling one-step validation, MAE/RMSE comparison, run metadata and explanatory notebooks. Notebook 03 reuses the nine recorded candidate results and performs one predetermined selected-order training fit per execution for numerical reproduction and diagnostics. The runner checks snapshot identity, loads only the development interval and validates calendar coverage before fitting.

**Files:** `scripts/build_naive_baseline.py`, `src/market_forecasting/dataset.py`, notebook 02, `tests/test_milestone2.py`, [Milestone 4](MILESTONE_4.md); ignored predictions and summary in `artifacts/naive_v1/`.

**Verification/results:** 502 validation forecasts; MAE **1.689494114 USD**, RMSE **2.589323812 USD** on the retrospective split-adjusted basis. These match the saved summary. Metric tests use known expected values. Notebook 02 checks the same metrics through explicit formulas and reports annual results, forecast errors and illustrative windows without fitting parameters or creating a separate experiment. No test-period score is available.

**Notebook evidence, 17 October 2026:** the twelve-section narrative covers bounded data access, session validity, chronological partitions, sequence construction, alignment, naive forecasting, rolling evaluation, MAE/RMSE, error analysis and limitations. All outputs remain inline. Original benchmark arrays, predictions and metadata are preserved, and recorded source hashes continue to identify the original runs. The shared execution and test totals are recorded under dataset construction above.

**ARIMA result:** selected (2,1,2); validation MAE 1.720200573 USD and RMSE 2.612846480 USD, approximately 1.82% worse MAE than naive. All nine candidates converged; starting-parameter warnings are retained. See [Milestone 4](MILESTONE_4.md) and [run metadata](arima_validation_v1.json).

**Added files:** `src/market_forecasting/arima.py`, `scripts/run_arima_benchmark.py`, `configs/arima.json`, `tests/test_arima.py`, notebook 03 and `requirements-arima-lock.txt`.

**ARIMA diagnostic evidence:** training Close gives ADF p = 0.704783 and KPSS p < 0.01; its first dollar difference gives ADF p approximately 8.15 × 10⁻¹⁶ and KPSS p > 0.10. These results are compatible with the specified first-difference treatment under the tests' assumptions; they do not establish constant variance or general stationarity. The AR and MA polynomials partly offset, so large individual coefficients do not establish strong net predictability. Training innovation plots and autocorrelation diagnostics examine remaining serial dependence and changing dispersion without expanding the candidate set or changing the selected order.

**Verification, 13 October 2026:** 17 tests passed. The ARIMA benchmark generated 502 forecasts on exactly the naive target dates. Notebook 03 executed successfully; dependency checks, snapshot identities and recorded training-source hashes were verified. Test-period performance remains uncomputed.

**Source provenance, 14 October 2026:** documentation-only source edits passed executable-syntax equivalence checks across 13 Python files after excluding docstrings, alongside 17 tests. Historical source hashes remain attached to the original runs rather than being rewritten; forecasting logic and numerical evidence were preserved.

**ARIMA verification, 17 October 2026:** 114 automated tests passed across the data, dataset/baseline, ARIMA, LSTM and feature suites (five existing Keras/NumPy deprecation warnings), and all 18 Notebook 03 code cells executed in a fresh kernel with 8 inline figures. One selected ARIMA(2,1,2) fit reproduced the recorded parameters and AIC 1756.733879471. All 502 rolling forecasts, target dates and benchmark metrics match the preserved evidence within the stated numerical tolerances. Trace events verify prediction before revealing each target and coefficient preservation during state updates. Bounded-loading and calendar checks protect the data path. The nine-candidate search is read from its recorded evidence rather than repeated, and no reserved-period price or performance is used. Notebook outputs remain inline; original experiment files and source-hash records are unchanged.

**Related comparisons:** residual LSTM results are recorded in [Milestone 5](MILESTONE_5.md), and controlled feature results in [Milestone 6](MILESTONE_6.md).

## 5. Stock-only LSTM

- [x] Freeze four configurations, three paired seeds and residual formulation before training.
- [x] Implement training-only scaling, reconstruction, early stopping and experiment recording.
- [x] Complete 12 internal-selection fits and three fixed-epoch full-development fits.
- [x] Verify alignment, leakage safeguards, checkpoint restoration and reload equivalence.
- [x] Record all runs, loss curves, benchmark comparisons and limitations.
- [x] Execute the 14-section architecture, preprocessing, optimisation and inference demonstration without retraining.
- [x] Verify all 15 checkpoints, target reconstruction, model selection and recorded loss histories.
- [x] Quantify input and correction distribution shifts without changing the model formulation.

**Objective/deliverables:** reproducible residual LSTM, controlled optimisation comparisons, locked configuration and external-validation results.

**Files:** `preprocessing.py`, `lstm.py`, experiment/report scripts, `tests/test_lstm.py`, search/selected configurations, LSTM environment lockfile, notebook 04, [design](MILESTONE_5_DESIGN.md), [technical summary](MILESTONE_5.md) and [results record](lstm_results_v1.json). Detailed histories, checkpoints and plots are local ignored artifacts.

**Results:** selected C (16 units, dropout 0.2, learning rate 0.001, batch 32), with 19 refit epochs. External MAE 1.705023 ± 0.013335 USD and RMSE 2.600560 ± 0.013501 USD across three seeds. Every seed is worse than naive on both metrics. All 15 fits completed; none failed or were replaced.

**Verification, 14 October 2026:** 26 tests passed, with five Keras/NumPy serialization deprecation warnings. All 15 complete run records and checkpoint reloads were verified; reloads reproduce predictions exactly in the recorded environment. The 502 external target dates match the benchmarks. Dependency checks, snapshot identities and source/configuration hashes matched. Notebook 04 executed all five code cells without errors. No extra fitting or 2025 forecasts occurred.

**Verification, 17 October 2026:** all 20 Notebook 04 code cells execute in order in a fresh kernel, retaining 11 inline figures and no notebook errors or warnings. The full automated suite passes 123 tests, with five existing Keras/NumPy serialization deprecation warnings. Tests include nine bounded-stage loading cases covering exclusion before decoding, snapshot integrity and calendar rejection. Both stages reproduce the saved scaling provenance, target counts and temporal alignment. All 15 loaded checkpoints reproduce recorded predictions, with maximum discrepancy 4.998 × 10⁻¹¹ USD from JSON rounding. Configuration C, median 19 epochs, 1,169 trainable parameters, 1,178 updates per final fit and all 502 external dates are verified. The runner uses the shared bounded loader; original numerical inputs and experimental evidence remain unchanged. No additional research training or reserved-period quote decoding occurs.

**Analytical findings:** end-of-epoch inference losses distinguish optimisation behaviour from the change in dollar-movement scale; internal naïve standardised MSE is 7.571391. Of 502 validation closes, 403 exceed the maximum fitting input price. Dollar-change SD rises from 1.015 USD in 2023 to 3.500 USD in 2024, while percentage-return SD is 3.051 and 3.307 percentage points. These observations motivate a controlled representation question, not a diagnosis of gate saturation or an approved model change. Complete calculations and interpretations are in [Notebook 04](../notebooks/04_lstm_experiments.ipynb).

**Limitations/next:** single internal holdout, three seeds, narrow search and changing input/volatility distributions. No definitive causal explanation for poor performance. Feature definitions and common eligibility were subsequently approved and implemented in Milestone 6; no further fitting is included in this milestone.

## 6. Controlled feature experiments

- [x] Approve and implement exact SMA/EMA/RSI/MACD definitions and SPY/XLK log returns.
- [x] Verify causal calculations, common warm-up, alignment and training-only scaling.
- [x] Complete three variants × three paired seeds at 19 fixed epochs.
- [x] Save per-run and aggregate results, annual metrics, paired differences and visualisations.
- [x] Verify saved checkpoints, experiment provenance, preserved Milestone 5 evidence and notebook execution.
- [x] Demonstrate indicator mathematics, real feature matrices, scaling and complete three-model inference in Notebook 05.
- [x] Verify feature ordering, shared input channels, causal prefix calculations and fixed-specification preflight rejection.
- [x] Interpret paired and annual results, feature redundancy, effective historical context and changing dollar scale.

**Objective/deliverables:** controlled comparison of price-only, technical-indicator and market-context residual LSTMs, plus interpretable feature visualisations.

**Files:** `features.py`, feature experiment/report scripts, feature tests and configuration, notebook 05, [approved specification](FEATURE_EXPERIMENT_PROPOSAL.md), [technical results](MILESTONE_6.md), [results JSON](feature_results_v1.json) and [verification record](feature_verification_v1.json). Local evidence is in `artifacts/features_v1/`.

**Results:** A/B/C mean MAE 1.702600/1.717304/1.715305 USD; mean RMSE 2.599713/2.609909/2.606177 USD. Each run uses 1,935 fitting examples and the original 502 validation dates. All nine fits completed; all have higher full-period errors than naive. No extra tuning or replacement runs occurred.

**Verification, 16 October 2026:** 35 tests passed, with five existing Keras/NumPy serialization deprecation warnings. Indicator seeds, first-valid dates, common eligibility, scaling and future perturbations are tested. All nine checkpoints reproduced predictions exactly, with 19 epochs and no validation monitoring. Ten training-source hashes and configuration hashes matched; dependency and snapshot checks passed. Notebook 05 executed six code cells without errors. The verification record confirms preservation of 74 protected files, including Milestone 5 evidence. No additional fitting or 2025 feature/evaluation work occurred. See [verification evidence](feature_verification_v1.json).

**Verification, 18 October 2026:** all 18 Notebook 05 code cells execute in order in a fresh kernel, retaining 13 inline figures and no notebook errors or warnings. The full suite passes 132 tests with five existing Keras/NumPy serialization deprecation warnings. Independent arithmetic verifies SMA windows, arithmetic EMA initialisation, Wilder RSI and edge cases, MACD identities and log returns. Prefix and future-perturbation checks verify causal construction; declared feature ordering, identical shared channels, 1,964 unique fitting input rows, 1,935 fitting labels and all 502 validation targets are checked. All nine checkpoints reproduce saved forecasts within 5.0 × 10⁻¹¹ USD of rounded JSON values; parameter counts, fixed settings, epoch-19 training losses, individual/aggregate/annual metrics and paired differences agree. The runner rejects configuration metadata inconsistent with implemented fixed formulas before data access or training. No feature values, scaler parameters, predictions or experiment evidence are changed, and no additional research fits or reserved-period quote decoding occurs.

**Analytical findings:** the seven technical features increase mean MAE by 0.014704 USD relative to the common-sample control. SPY/XLK decrease it by 0.001999 USD relative to B, with mixed seed and annual outcomes. Lower training losses for B/C do not yield better average validation accuracy. SMA20 and EMA12 distances have a fitting-row correlation of 0.957; relative MACD is algebraically determined by the two EMA distances. These findings describe redundancy, not individual importance or a reason to alter the fixed feature set. All calculations and interpretations are in [Notebook 05](../notebooks/05_feature_experiments.ipynb).

**Limitations/next:** grouped comparisons do not identify individual indicator importance; three seeds and a fixed training budget limit generalisation. Input dimensions change parameter counts, and indicators carry history preceding the 30 supplied feature rows: SMA50 spans 79 underlying closes across a 30-row sequence, while recursive states retain older information. Volume remains excluded. A return-target formulation would require separately approved preprocessing and controlled experiments; no formulation change or additional fit is implied. Final methodology awaits explicit approval before Milestone 7; no 2025 features or forecasts have been generated.

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
