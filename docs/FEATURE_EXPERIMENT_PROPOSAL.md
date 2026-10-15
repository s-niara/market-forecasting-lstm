# Milestone 6: proposed technical-feature experiments

**Status: feature proposal awaiting approval. No indicator implementation is included.** The price-only residual LSTM is complete; see [Milestone 5](MILESTONE_5.md). Technical feature engineering remains a central research contribution. ARIMA is a separate statistical benchmark and does not replace these experiments.

## Candidate indicators and parameters
All calculations use NVDA split-adjusted Close, denoted C(t), available after session t closes. These are fixed starting specifications, not validated optimal parameters. Periods count observed exchange sessions.

| Indicator | Proposed parameters | Causal definition | Candidate model input |
|---|---|---|---|
| SMA | 20 and 50 sessions | Arithmetic mean of the most recent n closes, including t; require n observations | C(t)/SMA(n,t) - 1 for each n |
| EMA | 12 and 26 sessions | Seed with the first n-close arithmetic mean; then E(t) = alpha*C(t) + (1-alpha)*E(t-1), alpha = 2/(n+1) | C(t)/EMA(n,t) - 1 for each n |
| RSI | 14 changes, Wilder smoothing | Seed average gain/loss with the first 14 price changes; recursively update each average with alpha = 1/14; RSI = 100 - 100/(1 + average gain/average loss) | RSI/100 |
| MACD | 12, 26, 9 | M(t) = EMA12(t) - EMA26(t); signal S(t) is a 9-period EMA of valid M values, seeded by their first nine-value mean; histogram H(t) = M(t) - S(t) | M(t)/C(t) and H(t)/C(t) |

The signal line remains available for charts and verification; it is omitted as a separate model input because M, S and H are exactly linearly dependent. MACD measures the separation of short and longer smoothing estimates, while RSI summarises the balance of positive and negative changes. Price-relative SMA, EMA and MACD inputs reduce dependence on NVDA's changing dollar scale without changing the dollar-price forecasting target.

For RSI, define zero loss with positive gain as 100; zero gain with positive loss as 0; both zero as 50. Explicit seeding avoids undocumented differences between technical-analysis libraries. An EMA seeded with the first observation is a legitimate alternative but would yield different early values. The proposed convention must be fixed before comparisons.

## Warm-up and temporal handling
- Compute in chronological order using only values at or before t. Do not use centred windows, negative shifts, future-fitted transformations or backfilling.
- Keep unavailable initial indicator values missing. SMA50 first becomes available on observation 50; EMA26 on observation 26; RSI14 on observation 15; MACD's signal on observation 34 under the specified seeds.
- Use the common complete-feature interval for all three LSTM variants, including the price-only control. With these definitions the first 49 feature rows are excluded. A subsequent 30-session window requires 30 complete feature rows; its earliest target is observation 80 of the source series. Determine dates and counts from the actual implementation rather than assuming them.
- This is the minimum mathematical warm-up, not proof of complete independence from EMA/RSI seeds. A longer initial burn-in is an alternative requiring a separately documented comparison.
- Continue indicator states through the training/validation boundary; do not reset on 1 January. Earlier validation observations may enter subsequent one-step inputs after they are observed. Do not calculate test-period features during development.
- Reject unexpected internal missing values or session misalignment rather than silently dropping rows or forward-filling across missing sessions. Half-days remain sessions.
- Fit feature and target scalers on training data only, once per experiment or fold. Use unique training feature rows to avoid overweighting observations repeated in overlapping windows. Transform validation with the fitted training scalers and invert target scaling before dollar-error calculation.

## Controlled comparisons
| Variant | Proposed inputs at each of 30 sessions | Purpose |
|---|---|---|
| A: price-only LSTM | NVDA Close | Control for the benefit of derived features |
| B: technical-indicator LSTM | A + two SMA distances, two EMA distances, RSI and two MACD inputs | Measure incremental value of technical features |
| C: multivariate LSTM | B + SPY and XLK one-session log Close returns | Measure incremental value of market and sector context |

SPY/XLK returns are proposed as log(C(t)/C(t-1)) on their frozen split-adjusted Close series, using the same session calendar. They are price-return context, not dividend-reinvested total returns. Both instruments contain NVDA; context is correlated and must not be described as independent information. Optional A + context would isolate market context without technical features, but is outside the initial three-run-family comparison.

The same target, effective training examples, validation target dates, 30-session sequence length, network structure, optimiser, epoch limit, batch size and stopping protocol must be used across variants. Input dimensionality changes trainable parameter counts; record these counts rather than claiming identical model capacity. The LSTM architecture is a later design decision and is not fixed by this proposal.

Proposed stochastic repetitions: seeds 42, 123 and 2026, paired across variants. Report each run plus mean and standard deviation; do not select only the best seed. Use dollar MAE as the primary comparison and dollar RMSE as secondary, with aligned error plots and separate 2023/2024 summaries. A small numerical difference alone does not establish statistical significance.

Architecture selection and early-stopping design follow the [Milestone 5 protocol](MILESTONE_5_DESIGN.md): fitting through 2021 and internal validation on 2022, followed by fitting through 2022 for external comparison. This supersedes the provisional three-fold tuning suggestion. The selected epoch budget and hyperparameters remain fixed across feature variants; no per-variant external-validation tuning is planned. Scalers and eligibility must be reconstructed for each fitting stage. The 2023–2024 period remains development comparison data and has already been examined for baselines; it is not an untouched final assessment.

Report naive and ARIMA scores alongside the variants on exactly the same validation target dates. The existing baselines retain their documented fitting procedures; do not rewrite earlier runs. If eligibility changes target dates, produce a separately labelled aligned comparison. ARIMA uses the full training history while LSTM examples use finite windows; disclose that distinction.

Freeze model and feature choices before the independent 2025 evaluation. Volume remains excluded until adjustment semantics are corroborated. No trading-profit or causal-market claims follow from forecast errors.

## Planned verification before experiments
- Compare indicator calculations with hand-computed examples, including RSI edge cases and EMA seeds.
- Verify first-valid positions, common warm-up handling and cross-boundary continuity.
- Perturb later prices and assert that earlier features and predictions do not change.
- Verify date alignment, train-only scaler fitting and target inversion.
- Verify identical eligible examples and paired seeds across the three variants.

These checks and all feature computations remain planned, not executed.

## Shared output formulation
All three variants use the approved residual dollar-correction formulation: inverse-transform the predicted correction, then add the unscaled origin NVDA Close. Fit correction scaling on training targets only. This keeps the price evaluation objective and reconstruction identical across feature sets without adding runs to Milestone 5.
