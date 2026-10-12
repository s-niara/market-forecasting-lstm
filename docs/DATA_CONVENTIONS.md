# Data conventions

## Source and price basis
Frozen Yahoo chart responses contain daily NVDA, SPY and XLK observations for 2015–2025. The [acquisition record](data_verification.json) identifies the snapshot vintage, checksums and initial quality checks.

The forecast target is **Yahoo split-adjusted Close in USD**. `adj_close` is retained as a reference field and is not the modelling target. No split or dividend multiplier is reapplied. Historical Close is retrospectively normalised and is not the original contemporaneous dollar quote. Dividend effects remain in price changes. This study does not reconstruct point-in-time corporate-action vintages or establish a deployable trading backtest.

## Session dates and coverage
Convert UTC timestamps to America/New_York before deriving daily session dates. Retain actual sessions, including half-days; do not fabricate weekend or holiday observations or forward-fill them as trading sessions.

The [calendar audit](calendar_audit_2026-10-12.json), using `pandas-market-calendars==5.5.0`, finds 2,766 expected dates with no missing or unexpected sessions, duplicates or ordering defects. NVDA uses NASDAQ; SPY and XLK use NYSE regular-equity sessions as a proxy for NYSE Arca. This is a calendar-library check, not an exchange-certified audit. Cross-instrument alignment alone does not prove complete session coverage.

The dataset builder assumes calendar coverage was audited separately; its ordering and numerical checks do not independently enforce exchange-session completeness.

## Forecast availability and temporal boundaries
| Partition | Target-date interval | Permitted use |
|---|---|---|
| Training | 2015-01-01–2022-12-31 | Exploration, parameter fitting and transformation fitting |
| Validation | 2023-01-01–2024-12-31 | Model comparison and bounded selection |
| Final test | 2025-01-01–2025-12-31 | Evaluation after selection is frozen |

The horizon is the next trading session, with the forecast formed after the origin session closes. All features must be available at that origin. Assign examples by target date. Input context may precede a partition boundary; future targets may not enter fitting.

The current builder truncates to the end of validation before numerical validation and window construction. No 2025 forecast targets or performance metrics are produced. Administrative quality checks have covered the full raw snapshot; the calendar audit inspects dates only. These checks must not be described as model evaluation.

## Windows, transformations and evaluation
The current development dataset contains unscaled, 30-session, single-feature Close windows. Thirty sessions is an initial setting, not a validated optimum. Scaling is unnecessary for the naive baseline; transformations required by later models must be fitted on training observations only.

The naive protocol is rolling one-step forecasting: use the preceding session's observed Close for each target. Earlier validation observations may enter later forecast inputs as they become available. This does not constitute parameter fitting on validation data or a single long-horizon forecast. See [the technical summary](MILESTONE_2.md) for executed results.

## Unresolved volume and portability constraints
Keep vendor volumes unchanged and exclude volume features pending independent corroboration of adjustment semantics. Split-date continuity alone does not establish the provider's volume methodology. No verified original-tape comparison is available. Daily Close multiplied by volume must not be represented as exact traded dollar turnover.

Other technical-feature definitions remain outstanding. SPY and XLK contain NVDA and represent correlated context, not independent information. Short histories, exchange calendars and currencies require explicit treatment before claiming general portability.

## Reference provenance
- [NVIDIA 2021 split announcement](https://nvidianews.nvidia.com/news/nvidia-announces-four-for-one-stock-split-pending-stockholder-approval-at-annual-meeting-set-for-june-3)
- [NVIDIA 2024 SEC filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm)
- [Yahoo adjusted-close explanation](https://help.yahoo.com/kb/SLN28256.html): a search excerpt was available during the initial audit; direct retrieval was rate-limited.
- [Calendar-library scheduling documentation](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html)

Another vendor's adjustment description was insufficient to certify Yahoo volume semantics. No verification claim is based on it.
