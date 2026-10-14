# Milestone 4: naive and ARIMA benchmarks

## Scope and protocol
The naive and ARIMA benchmarks are implemented for next-session Yahoo split-adjusted Close in USD. They support the planned LSTM feature comparisons; neither replaces technical feature engineering. Existing target conventions and 2025 safeguards are unchanged.

ARIMA uses all 2,014 training observations from 2015-01-02 to 2022-12-30. Nine nonseasonal orders combine p and q in {0,1,2} with d=1, no drift, and stationary/invertible ARMA constraints. The optimiser allows 500 iterations. The lowest finite training AIC among converged candidates selects the model, with deterministic lower-complexity tie-breaking. Validation errors do not select the order.

First differencing is a bounded benchmark assumption aimed at modelling changes in price level. It does not establish stationarity, constant volatility or structural stability. No drift avoids imposing constant expected dollar growth. Alternatives include drift, transformed targets, wider searches and periodic parameter refitting; these are not part of this run. No scaler or log-target transformation is fitted.

For each validation session, issue the forecast before observing its target Close. Then update the filtered state with that observed value using `extend`, without re-estimating coefficients. Fixed parameters provide a transparent comparison with later models fitted on training data. Updating the filtered state is distinct from refitting on validation observations.

## Naive benchmark
The naive forecast equals the final Close in each input window. Only that value is required by the model; the shared 30-session representation supports later comparisons. [Milestone 3](MILESTONE_3.md) documents the windows and target-date partitioning. Evaluation proceeds one session at a time as observations become available.

## Executed result
The selected order is **ARIMA(2,1,2)**, with training AIC **1756.733879471**. All nine candidates converged. Orders (1,1,2) and (2,1,2) emitted starting-parameter stationarity/invertibility warnings; statsmodels substituted zero starting values. These warnings are preserved rather than suppressed. Forecasting emitted no warnings.

| Model | Validation forecasts | MAE (USD) | RMSE (USD) |
|---|---|---|---|
| Previous-session Close | 502 | 1.689494114 | 2.589323812 |
| ARIMA(2,1,2) | 502 | 1.720200573 | 2.612846480 |

Both models use target dates 2023-01-03 through 2024-12-31. ARIMA MAE is approximately 1.82% higher than naive MAE. Training AIC ranks in-sample fit with a complexity penalty; it does not guarantee better out-of-sample forecasts. This result is retained without extending the search to obtain a favourable comparison. Dollar-scale regime changes and fixed coefficients are limitations; no superiority or statistical-significance claim is made. No 2025 performance was computed.

## Evidence and reproduction
The original naive run is preserved under `artifacts/naive_v1/`. Its `summary.json` records configuration and snapshot hashes, and its validation errors match the naive comparator in the ARIMA run. [Notebook 02](../notebooks/02_dataset_and_naive_baseline.ipynb) recomputes development windows and naive metrics in memory. `scripts/build_naive_baseline.py` creates the fixed `artifacts/naive_v1/` destination and refuses overwriting; a new artifact run requires a separately versioned destination with a recorded rationale. These fixed-path scripts are not a general experiment-versioning interface. Executed notebook copies remain local under `artifacts/`.

[Aggregate run metadata](arima_validation_v1.json) records candidate AIC values, warnings, fitted parameters, environment, settings, snapshot identity and code hashes. Per-session predictions and the original summary are local under `artifacts/arima_v1/`; raw vendor data and generated predictions remain excluded from Git.

Install the environment using `requirements-arima-lock.txt`, followed by `python -m pip install -e .`. From the repository root, run `python scripts/check_snapshot.py` and `python -m pytest -q`. To reproduce into a new directory, run `python scripts/run_arima_benchmark.py --run-name arima_reproduction_01`. Existing run directories are never overwritten. Aggregate results may vary slightly across numerical-library/platform versions; the original environment is recorded.

[Notebook 03](../notebooks/03_arima_benchmark.ipynb) displays the saved run, candidate selection and aligned forecasts. It does not refit or evaluate test data. Source-code hashes identify the exact source bytes used for the original run. Subsequent docstring/comment additions change these hashes without changing forecasting logic; the historical metadata is preserved rather than rewritten. Documentation-only equivalence checks and current verification are recorded in [PROGRESS.md](PROGRESS.md).

## Verification
The original audit/dataset/naive implementation reported 12 passing tests, matching snapshot checksums, successful execution of notebooks 01–02 and consistent dependencies. The naive metrics are checked against known numerical examples. Visual overlap between actual and predicted prices does not establish predictive skill; dollar errors depend on the price scale and must be compared on identical target dates and forecast availability.

At completion of the ARIMA implementation, 17 automated tests passed. ARIMA-specific tests verify equivalence between ARIMA(0,1,0) without drift and the naive forecast, forecast-before-observation ordering, agreement with an independent `append(refit=False)` update, exclusion of post-validation values, invariance of earlier predictions to later target changes, training-only parameter selection, and rejection/recording of nonconverged or failed fits.

## References
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Statsmodels state extension](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.extend.html)

The next modelling stage is the stock-only LSTM design. Indicator implementation awaits approval of the [Milestone 6 proposal](FEATURE_EXPERIMENT_PROPOSAL.md).
