# Data conventions

## Source and price basis
Frozen Yahoo chart responses contain daily NVDA, SPY and XLK observations for 2015–2025. The [acquisition record](data_verification.json) identifies the snapshot vintage, checksums and initial quality checks.

The forecast target is **Yahoo split-adjusted Close in USD**. `adj_close` is retained as a reference field and is not the modelling target. No split or dividend multiplier is reapplied. Historical Close is retrospectively normalised and is not the original contemporaneous dollar quote. Dividend effects remain in price changes. This study does not reconstruct point-in-time corporate-action vintages or establish a deployable trading backtest.

## Session dates and coverage
Convert UTC timestamps to America/New_York before deriving daily session dates. Retain actual sessions, including half-days; do not fabricate weekend or holiday observations or forward-fill them as trading sessions.

The [calendar audit](calendar_audit_2026-10-12.json), using `pandas-market-calendars==5.5.0`, finds 2,766 expected dates with no missing or unexpected sessions, duplicates or ordering defects. NVDA uses NASDAQ; SPY and XLK use NYSE regular-equity sessions as a proxy for NYSE Arca. This is a calendar-library check, not an exchange-certified audit. Cross-instrument alignment alone does not prove complete session coverage.

The development dataset builder assumes calendar coverage was audited separately; its ordering and numerical checks do not independently enforce exchange-session completeness. Final fit-only preparation and forecast-origin preparation additionally require complete exchange-session history.

## Forecast availability and temporal boundaries
| Partition | Target-date interval | Permitted use |
|---|---|---|
| Training | 2015-01-01–2022-12-31 | Exploration, parameter fitting and transformation fitting |
| Validation | 2023-01-01–2024-12-31 | Model comparison and bounded selection |
| Final refit | 2015-01-01–2024-12-31 | Fit frozen specifications after development selection |
| Final test | 2025-01-01–2025-12-31 | Evaluation only after artifact freeze and separate authorisation |

The horizon is the next trading session, with the forecast formed after the origin session closes. All features must be available at that origin. Assign examples by target date. Input context may precede a partition boundary; future targets may not enter fitting.

The current builder truncates to the end of validation before numerical validation and window construction. No 2025 forecast targets or performance metrics are produced. Administrative quality checks have covered the full raw snapshot; the calendar audit inspects dates only. These checks must not be described as model evaluation.

## Bounded exploratory access
Notebook 01 uses `load_training_panel()`, which passes explicit inclusive start/end dates to the reusable Yahoo loader. Only permitted OHLCV and Adjusted Close values are decoded and validated. The reader examines timestamps through the first later session to locate the boundary; excluded quote values and current-price metadata are not deserialised. It supports the frozen chronological scalar-array schema and rejects ambiguous required fields. It does not certify ordering or completeness in an unread future tail.

Opaque snapshot checksums establish identity, separately from numerical validation. `price_quality()` supplies the diagnostic counts used by `validate_prices()`; an independent calendar audit remains required because matching instrument indexes can share the same missing session. The unbounded snapshot API is available for administrative use; Notebook 01 always uses explicit bounds.

Notebook 02 uses explicit development bounds ending 2024-12-31. Snapshot provenance and an independent NASDAQ session audit precede dataset construction. The same bounded loader restricts quote decoding; the dataset builder additionally restricts the interval before numerical validation and window construction. No reserved-period price is needed for this path.

## Windows, transformations and evaluation
The price-only development builder accepts a pandas Series indexed by chronological session dates and produces unscaled, 30-session, single-feature Close windows. A DataFrame does not satisfy this one-dimensional input contract. Each window ends at its forecast origin and its target is the immediately following observed session. The representation contains 1,984 training and 502 validation examples with shape `(2486, 30, 1)`. Thirty sessions defines the common lookback, not a validated optimum. Scaling is unnecessary for the naive baseline; model-specific transformations must be fitted on their permitted training observations only.

The naive protocol is rolling one-step forecasting: use the preceding session's observed Close for each target. Earlier validation observations may enter later forecast inputs as they become available. This does not constitute parameter fitting on validation data or a single long-horizon forecast. See [Milestone 3](MILESTONE_3.md) for dataset construction and [Milestone 4](MILESTONE_4.md) for executed forecast results.

## Unresolved volume and portability constraints
Keep vendor volumes unchanged and exclude volume features pending independent corroboration of adjustment semantics. Split-date continuity alone does not establish the provider's volume methodology. No verified original-tape comparison is available. Daily Close multiplied by volume must not be represented as exact traded dollar turnover.

The approved SMA, EMA, RSI, MACD and context-return definitions are implemented in Milestone 6. SPY and XLK contain NVDA and represent correlated context, not independent information. Short histories, exchange calendars and currencies require explicit treatment before claiming general portability.

## Reference provenance
- [NVIDIA 2021 split announcement](https://nvidianews.nvidia.com/news/nvidia-announces-four-for-one-stock-split-pending-stockholder-approval-at-annual-meeting-set-for-june-3)
- [NVIDIA 2024 SEC filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm)
- [Yahoo adjusted-close explanation](https://help.yahoo.com/kb/SLN28256.html): a search excerpt was available during the initial audit; direct retrieval was rate-limited.
- [Calendar-library scheduling documentation](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html)

Another vendor's adjustment description was insufficient to certify Yahoo volume semantics. No verification claim is based on it.

## Residual LSTM transformations
The network predicts a standardised dollar correction, defined as next-session Close minus forecast-origin Close. Inverse-transform the correction and add the unscaled origin Close for USD evaluation. A zero-dollar correction reproduces the naive forecast; zero standardised output instead represents the fitted mean correction.

Input standardisation uses unique observations occurring in fitting windows; correction standardisation uses fitting labels. Both use population standard deviation, with scale one for a constant series. No clipping or future-value imputation is performed. Selection fits 1,733 targets through 2021-12-31 and scores 251 targets in 2022. The input scaler uses 1,762 unique observations through 2021-12-30. After selection, fresh scalers and weights use 1,984 targets through 2022-12-30, with 2,013 input observations through 2022-12-29. External evaluation covers the original 502 dates. Scalers remain fixed during each evaluation phase. See [the design](MILESTONE_5_DESIGN.md) and [recorded statistics](lstm_results_v1.json).

## Milestone 6 common feature eligibility
The [approved indicator definitions](FEATURE_EXPERIMENT_PROPOSAL.md) are implemented with arithmetic EMA seeds and Wilder RSI. The first 49 rows remain missing; no backfill is used. All variants share the complete interval beginning 2015-03-16 and 30-session windows, yielding 1,935 fitting targets from 2015-04-28 and the original 502 validation dates. Fit each feature scaler on unique fitting input rows through 2022-12-29 and the common correction scaler on fitting labels through 2022-12-30. Continue feature states across boundaries without reinitialising them. Reject internal gaps and mismatched instrument sessions. No test-period features are computed. [Milestone 6](MILESTONE_6.md) records exact statistics and results.

## Final fit-only preparation and frozen-origin inference
Final fitting uses the unchanged snapshots and conventions over 2015-01-01–2024-12-31, yielding 2,516 observed sessions. All three LSTM variants retain the common 49-row warm-up: 2,437 target labels from 2015-04-28 through 2024-12-31 and 2,466 unique scaler-fitting input rows from 2015-03-16 through 2024-12-30. The last fitting target is not added to the input scaler. Each feature has its own population-standard-deviation scaler; all variants share the same fitted dollar-correction target statistics. No new validation partition is fabricated for final fitting.

`prepare_final_feature_dataset()` reuses the existing feature/window construction with an all-fitting mask. `prepare_origin_input()` validates frozen feature ordering and scaler metadata, selects the observed prefix before numerical conversion, audits complete aligned sessions, and calls the authoritative indicator calculations. It returns one `(1,30,F)` tensor and the origin Close without requiring the next-session target. Recursive indicators retain their original seeds; passing only the last 30 raw closes is insufficient.

The bounded loader never decodes quotes beyond 2024-12-31 during fit or verification. Origin and ARIMA sequencing interfaces default to pre-test limits. A caller may supply an explicit permitted-history boundary for evaluation; the real evaluation runner cannot do so until its separate release gates are authorised. Reload checks use origins no later than 2024-12-30; the final ARIMA state is inspected without issuing its next, reserved-period forecast. Synthetic checks and pre-test state continuation verify mechanics without constituting final accuracy evidence.

Final artifacts bind scalers, feature names/order, target formulation, model state and source/data identities in the [freeze manifest](../artifacts/final_training_v1/manifest.json). Future rolling input updates and ARIMA state filtering must not update LSTM weights, fitted scalers or ARIMA coefficients. The expected 250 test sessions, common target dates and historical context will be verified only when evaluation is authorised.

## Independent evaluation observation sequence
The [evaluation configuration](../configs/final_evaluation_v1.json) binds the pinned training manifest to the unchanged scientific protocol and an inactive execution-authorisation flag. A separate inactive code-level gate prevents configuration changes alone from releasing actual test quotes. Synthetic tests use synthetic sessions, and the `dry-run` command limits its real observations and targets to 2024.

The evaluator requires identical, complete consecutive exchange sessions for NVDA, SPY and XLK. It does not intersect calendars, drop missing dates or fill absent observations. Each iteration constructs the forecast-origin tensor from the observed prefix, records all model/seed predictions, then loads only the corresponding next-session observations. The observed Close becomes the next origin after errors are recorded. Price-free session coverage and source identity are distinct from this sequential quote decoding.

The canonical evaluation table has one row per target date, common origin/target prices and eleven prediction columns: naive, ARIMA and three seeds for each of A/B/C. Signed errors are prediction minus actual. Family means use individually calculated seed MAE and RMSE; they are not metrics of averaged predictions. Paired comparisons retain matching dates and seeds. Percentage skill is undefined when the naive denominator is zero. Missing columns, nonfinite values, mismatched origins and incomplete session sequences cause explicit failure.

The eventual single evaluation record includes observed NVDA/SPY/XLK closes, predictions, signed errors, seed/family metrics, benchmark-relative skill, paired differences, supplementary bootstrap intervals and verification status. It also records the original frozen training identity and the separate evaluation implementation identity. No such independent-test record is generated before release authorisation.
