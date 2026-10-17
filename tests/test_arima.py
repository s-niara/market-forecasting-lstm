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



def test_rolling_trace_records_forecast_before_each_update():
    calls = []
    class State:
        params = np.array([.3, .4])
        def __init__(self, last): self.last = last
        def forecast(self, steps):
            assert steps == 1
            calls.append(("forecast", self.last))
            return np.array([self.last + 1])
        def extend(self, observed):
            calls.append(("update", observed[0]))
            return State(observed[0])
    trace = []
    result = arima.rolling_one_step(State(10.), [20., 30.], trace=trace)
    np.testing.assert_array_equal(result, [11., 21.])
    assert calls == [("forecast", 10.), ("update", 20.), ("forecast", 20.), ("update", 30.)]
    assert [(r["step"], r["event"]) for r in trace] == [(0, "forecast"), (0, "update"),
                                                      (1, "forecast"), (1, "update")]
    assert all("observed_close" not in r for r in trace if r["event"] == "forecast")
    assert [r["observed_close"] for r in trace if r["event"] == "update"] == [20., 30.]
    assert all(r["parameters"] == [.3, .4] for r in trace)
    # Trace snapshots must not retain mutable aliases to the model's parameter vector.
    State.params[0] = 99.
    assert trace[0]["parameters"] == [.3, .4]


def test_rolling_trace_preserves_exact_forecasts_and_fixed_coefficients():
    close, _, _ = fixture_data()
    fitted, _ = arima.select_training_model(close.iloc[:70], [[1, 1, 0]])
    original_params = fitted.params.copy()
    expected = arima.rolling_one_step(fitted, close.iloc[70:75])
    trace = []
    actual = arima.rolling_one_step(fitted, close.iloc[70:75], trace=trace)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fitted.params, original_params)
    assert len(trace) == 10
    for event in trace:
        np.testing.assert_array_equal(event["parameters"], original_params)


def test_training_aic_tie_break_and_nonfinite_parameters(monkeypatch):
    class Result:
        mle_retvals = {"converged": True}
        def __init__(self, order):
            self.order = order
            self.aic = -999. if order == (2, 1, 0) else 10.
            self.params = np.array([np.inf if order == (2, 1, 0) else .5])
    class Model:
        def __init__(self, values, order, **kwargs): self.order = order
        def fit(self, **kwargs): return Result(self.order)
    monkeypatch.setattr(arima, "ARIMA", Model)
    fitted, records = arima.select_training_model(np.arange(60, dtype=float) + 10,
                                                  [[2, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]])
    assert not records[0]["eligible"]
    # Equal AIC favours smaller p+q, then the lexicographically smaller order.
    assert fitted.order == (0, 1, 1)


def load_arima_script():
    """Import the runner without executing its command-line entry point."""
    import importlib.util
    from pathlib import Path
    root = Path(arima.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("arima_runner_test", root / "scripts/run_arima_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arima_repository(tmp_path, *, missing_session=False, future_marker=False):
    """Build a synthetic 2020 repository with a matching acquisition checksum."""
    import hashlib
    import json
    root = tmp_path / "synthetic_repository"
    for directory in ("configs", "data/raw", "docs", "artifacts"):
        (root / directory).mkdir(parents=True)
    config = dict(start="2020-01-02", train_end="2020-01-06", validation_end="2020-01-10",
                  test_end="2020-01-17", target_ticker="NVDA", price_convention="split_adjusted_close",
                  initial_sequence_length=2, forecast_horizon_trading_days=1)
    settings = dict(orders=[[0, 1, 0]], trend="n", selection="training_aic", refit=False,
                    maxiter=500, calendar="NASDAQ")
    (root / "configs/nvda.json").write_text(json.dumps(config))
    (root / "configs/arima.json").write_text(json.dumps(settings))
    dates = pd.bdate_range("2020-01-02", "2020-01-13")
    close = list(np.arange(11., 11. + len(dates)))
    quote = {"open": [v - .5 for v in close], "high": [v + 1 for v in close],
             "low": [v - 1 for v in close], "close": close, "volume": [100] * len(dates)}
    adjusted = [v - .25 for v in close]
    timestamps = [int((date.tz_localize("America/New_York") + pd.Timedelta(hours=16)).timestamp())
                  for date in dates]
    if missing_session:
        position = dates.get_loc(pd.Timestamp("2020-01-07"))
        timestamps.pop(position)
        for values in quote.values(): values.pop(position)
        adjusted.pop(position)
    if future_marker:
        quote["close"][-1] = "UNREAD_FUTURE_QUOTE"
    payload = {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"}, "timestamp": timestamps,
        "indicators": {"quote": [quote], "adjclose": [{"adjclose": adjusted}]}}]}}
    snapshot = root / "data/raw/NVDA_yahoo_snapshot.json"
    snapshot.write_text(json.dumps(payload))
    acquisition = {"retrieved_at_utc": "2020-02-01T00:00:00+00:00", "series": {
        "NVDA": {"sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}}}
    (root / "docs/data_verification.json").write_text(json.dumps(acquisition))
    return root


def test_runner_bounds_inputs_and_records_sources_without_model_fitting(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import market_forecasting.data as data_module
    root = arima_repository(tmp_path, future_marker=True)
    script = load_arima_script()
    loader, number = script.load_yahoo_snapshot, data_module._number
    bounds, experiments = [], []
    def bounded_loader(path, *, start, end):
        bounds.append((start, end))
        return loader(path, start=start, end=end)
    def selected_number(token, key):
        assert b"UNREAD_FUTURE_QUOTE" not in token
        return number(token, key)
    def fake_experiment(close, config, settings):
        experiments.append(close.copy())
        predictions = pd.DataFrame({"origin_date": close.index[2:-1], "target_date": close.index[3:],
            "actual_close": close.iloc[3:].to_numpy(), "arima_prediction": close.iloc[2:-1].to_numpy(),
            "naive_prediction": close.iloc[2:-1].to_numpy()})
        summary = {"selected_order": [0, 1, 0], "validation_metrics": {
            "arima": {"n": 4, "mae_usd": 1., "rmse_usd": 1.},
            "naive": {"n": 4, "mae_usd": 1., "rmse_usd": 1.}}, "mae_improvement_pct": 0.}
        return SimpleNamespace(predictions=predictions, summary=summary)
    monkeypatch.setattr(script, "load_yahoo_snapshot", bounded_loader)
    monkeypatch.setattr(data_module, "_number", selected_number)
    monkeypatch.setattr(script, "run_arima_experiment", fake_experiment)
    script.main([], root=root)
    assert bounds == [("2020-01-02", "2020-01-10")]
    assert len(experiments) == 1 and len(experiments[0]) == 7
    assert experiments[0].index.max() == pd.Timestamp("2020-01-10")
    np.testing.assert_array_equal(experiments[0].to_numpy(), np.arange(11., 18.))
    output = root / "artifacts/arima_v1"
    recorded = json.loads((output / "summary.json").read_text())
    assert recorded["selected_order"] == [0, 1, 0]
    assert recorded["calendar_audit"]["missing"] == []
    assert "src/market_forecasting/data.py" in recorded["source_sha256"]
    table = pd.read_json(output / "validation_predictions.json")
    np.testing.assert_array_equal(table.arima_prediction, [13., 14., 15., 16.])


def test_runner_refuses_changed_snapshot_before_loading_or_fitting(tmp_path, monkeypatch):
    root = arima_repository(tmp_path)
    snapshot = root / "data/raw/NVDA_yahoo_snapshot.json"
    snapshot.write_text(snapshot.read_text() + " ")
    script = load_arima_script()
    def forbidden(*args, **kwargs): raise AssertionError("Identity must be checked first")
    monkeypatch.setattr(script, "load_yahoo_snapshot", forbidden)
    monkeypatch.setattr(script, "run_arima_experiment", forbidden)
    with pytest.raises(SystemExit, match="[Cc]hecksum"):
        script.main([], root=root)
    assert not (root / "artifacts/arima_v1").exists()


def test_runner_refuses_missing_session_before_fitting(tmp_path, monkeypatch):
    root = arima_repository(tmp_path, missing_session=True)
    script = load_arima_script()
    def forbidden(*args, **kwargs): raise AssertionError("Calendar must be checked first")
    monkeypatch.setattr(script, "run_arima_experiment", forbidden)
    with pytest.raises(SystemExit, match="coverage"):
        script.main([], root=root)
    assert not (root / "artifacts/arima_v1").exists()


@pytest.mark.parametrize("fault", ["unexpected", "duplicate_dates", "ordered"])
def test_runner_refuses_other_calendar_defects_before_fitting(tmp_path, monkeypatch, fault):
    root = arima_repository(tmp_path)
    script = load_arima_script()
    report = {"missing": [], "unexpected": [], "duplicate_dates": 0, "ordered": True}
    report[fault] = {"unexpected": ["2020-01-04"], "duplicate_dates": 1, "ordered": False}[fault]
    def forbidden(*args, **kwargs): raise AssertionError("Calendar must be checked first")
    monkeypatch.setattr(script, "audit_sessions", lambda *args, **kwargs: report)
    monkeypatch.setattr(script, "run_arima_experiment", forbidden)
    with pytest.raises(SystemExit, match="coverage"):
        script.main([], root=root)
    assert not (root / "artifacts/arima_v1").exists()


def test_runner_refuses_overwrite_before_source_access(tmp_path, monkeypatch):
    root = arima_repository(tmp_path)
    output = root / "artifacts/arima_v1"
    output.mkdir()
    marker = output / "preserved.txt"
    marker.write_text("preserve experiment")
    script = load_arima_script()
    def forbidden(*args, **kwargs): raise AssertionError("Existing run must be refused first")
    monkeypatch.setattr(script, "snapshot_provenance", forbidden)
    monkeypatch.setattr(script, "load_yahoo_snapshot", forbidden)
    monkeypatch.setattr(script, "run_arima_experiment", forbidden)
    with pytest.raises(SystemExit, match="already exists"):
        script.main([], root=root)
    assert marker.read_text() == "preserve experiment"
    assert list(output.iterdir()) == [marker]


def test_runner_rejects_path_traversal_in_run_name(tmp_path):
    root = arima_repository(tmp_path)
    with pytest.raises(SystemExit):
        load_arima_script().main(["--run-name", "../outside"], root=root)
    assert not list((root / "artifacts").iterdir())
