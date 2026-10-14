"""ARIMA protocol checks using synthetic prices and controlled fitting failures."""
import numpy as np
import pandas as pd
import pytest
from market_forecasting import arima


def fixture_data():
    """Return 100 seeded synthetic prices, split settings and one candidate order."""
    # Reserve 70 observations for fitting, 20 for validation and 10 for future-data tests.
    # Business-day labels are synthetic and do not imply exchange-calendar coverage.
    dates = pd.bdate_range("2022-01-03", periods=100)
    values = 30 + np.cumsum(np.random.default_rng(42).normal(0, 0.15, len(dates)))
    config = dict(price_convention="split_adjusted_close", start=str(dates[0].date()),
                  train_end=str(dates[69].date()), validation_end=str(dates[89].date()),
                  test_end=str(dates[-1].date()), forecast_horizon_trading_days=1,
                  initial_sequence_length=30)
    settings = dict(selection="training_aic", refit=False, trend="n", orders=[[1, 1, 0]], maxiter=500)
    return pd.Series(values, index=dates), config, settings


def test_random_walk_matches_naive():
    """Check no-drift ARIMA(0,1,0) reproduces previous-close forecasts and dates."""
    close, config, settings = fixture_data()
    settings["orders"] = [[0, 1, 0]]
    result = arima.run_arima_experiment(close, config, settings)
    np.testing.assert_allclose(result.predictions.arima_prediction,
                               result.predictions.naive_prediction, atol=1e-8)
    assert len(result.predictions) == 20
    assert result.predictions.origin_date.iloc[0] == close.index[69]
    assert result.predictions.target_date.iloc[0] == close.index[70]


def test_future_targets_cannot_change_earlier_forecasts_or_fit():
    """Verify later observations cannot alter earlier predictions or training selection."""
    close, config, settings = fixture_data()
    original = arima.run_arima_experiment(close, config, settings)
    changed = close.copy()
    changed.iloc[80:90] += 5
    changed.iloc[90:] = np.nan
    altered = arima.run_arima_experiment(changed, config, settings)
    # Target at position 80 is also unknown when its own forecast is issued.
    np.testing.assert_allclose(original.predictions.arima_prediction.iloc[:11],
                               altered.predictions.arima_prediction.iloc[:11], rtol=0, atol=0)
    assert original.summary["parameters"] == altered.summary["parameters"]
    assert original.summary["candidates"] == altered.summary["candidates"]
    test_changed = close.copy()
    test_changed.iloc[90:] = np.nan
    other = arima.run_arima_experiment(test_changed, config, settings)
    pd.testing.assert_frame_equal(original.predictions, other.predictions)


def test_forecast_precedes_observation_and_parameters_stay_fixed():
    """Compare extend-based forecasts with append(refit=False) and preserve parameters."""
    close, _, _ = fixture_data()
    fitted, _ = arima.select_training_model(close.iloc[:70], [[1, 1, 0]])
    params = fitted.params.copy()
    expected_first = float(fitted.forecast(1)[0])
    expected_second = float(fitted.append([close.iloc[70]], refit=False).forecast(1)[0])
    actual = arima.rolling_one_step(fitted, close.iloc[70:72])
    np.testing.assert_allclose(actual, [expected_first, expected_second], atol=1e-8)
    np.testing.assert_array_equal(fitted.params, params)


def test_selection_rejects_nonconverged_fit_and_records_failure(monkeypatch):
    """Use controlled fits to check eligibility, failure recording and all-failed rejection."""
    class Result:
        params = np.array([1.])
        def __init__(self, p):
            self.aic = -999 if p == 0 else 5
            self.mle_retvals = {"converged": p != 0}
    class Model:
        def __init__(self, values, order, **kwargs): self.p = order[0]
        def fit(self, **kwargs):
            if self.p == 2: raise ValueError("synthetic fitting failure")
            return Result(self.p)
    monkeypatch.setattr(arima, "ARIMA", Model)
    best, rows = arima.select_training_model(np.arange(1., 61.), [[0,1,0],[1,1,0],[2,1,0]])
    assert best.aic == 5
    assert not rows[0]["eligible"] and rows[1]["eligible"]
    assert "synthetic fitting failure" in rows[2]["error"]
    with pytest.raises(RuntimeError, match="No converged"):
        arima.select_training_model(np.arange(1., 61.), [[0,1,0]])


def test_unsupported_protocol_rejected():
    """Reject refitting requests instead of silently changing the evaluation protocol."""
    close, config, settings = fixture_data()
    settings["refit"] = True
    with pytest.raises(ValueError, match="fixed parameters"):
        arima.run_arima_experiment(close, config, settings)
