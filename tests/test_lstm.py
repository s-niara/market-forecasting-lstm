"""Residual scaling, chronology, checkpoint and selection checks without neural training."""
import json
import numpy as np
import pandas as pd
import pytest
from market_forecasting.preprocessing import (Standardisation, prepare_residual_data,
                                            reconstruct_prices)
from market_forecasting.lstm import (configure_runtime, build_model, predict_standardised,
                                    EarlyStopState, choose_configuration, train_model)

configure_runtime()


def sample():
    """Return curved synthetic prices with separate fitting, validation and future blocks."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(10 + np.arange(100) ** 2 / 50, index=dates)
    cfg = dict(start=str(dates[0].date()), train_end=str(dates[79].date()),
               validation_end=str(dates[89].date()), test_end=str(dates[-1].date()),
               price_convention="split_adjusted_close", forecast_horizon_trading_days=1,
               initial_sequence_length=30)
    return close, cfg, str(dates[59].date()), str(dates[79].date())


def test_scaling_unique_inputs_and_correction_alignment():
    close, cfg, fit_end, end = sample()
    data = prepare_residual_data(close, cfg, fit_end, end)
    # Targets 30..59 use input dates 0..58 exactly once for fitting statistics.
    assert data.provenance["input_unique_rows"] == 59
    assert data.input_scaler.mean == pytest.approx(close.iloc[:59].mean())
    assert data.input_scaler.mean != pytest.approx(data.dataset.X[data.fit_mask].mean())
    expected = close.iloc[30:60].to_numpy() - close.iloc[29:59].to_numpy()
    assert data.correction_scaler.mean == pytest.approx(expected.mean())
    assert data.X.shape == (50, 30, 1)
    assert data.dataset.metadata.target_date.iloc[30] == close.index[60]
    assert data.dataset.metadata.origin_date.iloc[30] == close.index[59]


def test_future_values_do_not_change_fit_or_earlier_inputs():
    close, cfg, fit_end, end = sample()
    a = prepare_residual_data(close, cfg, fit_end, end)
    altered = close.copy()
    altered.iloc[65:80] += 1000
    altered.iloc[80:] = np.nan
    b = prepare_residual_data(altered, cfg, fit_end, end)
    assert a.provenance == b.provenance
    np.testing.assert_array_equal(a.X[:36], b.X[:36])
    np.testing.assert_array_equal(a.y[a.fit_mask], b.y[b.fit_mask])
    only_future = close.copy()
    only_future.iloc[80:] = np.nan
    c = prepare_residual_data(only_future, cfg, fit_end, end)
    np.testing.assert_array_equal(a.X, c.X)
    np.testing.assert_array_equal(a.y, c.y)


def test_reconstruction_zero_correction_and_error_equivalence():
    scaler = Standardisation.fit([1., 2., 3.])
    origins = np.array([30., 40.])
    # Zero standardised output is the fitting mean; zero dollars needs transformation.
    np.testing.assert_array_equal(reconstruct_prices(origins, np.zeros(2), scaler), origins + 2)
    np.testing.assert_allclose(reconstruct_prices(origins, scaler.transform(np.zeros(2)), scaler), origins)
    predicted_change, actual_change = np.array([.5, -2.]), np.array([1., 1.])
    prices = reconstruct_prices(origins, scaler.transform(predicted_change), scaler)
    np.testing.assert_allclose(prices - (origins + actual_change), predicted_change - actual_change)
    with pytest.raises(ValueError): reconstruct_prices(origins, np.zeros((2, 1)), scaler)
    with pytest.raises(ValueError): reconstruct_prices(origins, np.array([np.nan, 0]), scaler)


def test_constant_scaler_and_invalid_boundaries():
    assert Standardisation.fit([2., 2.]).scale == 1
    close, cfg, _, _ = sample()
    with pytest.raises(ValueError): prepare_residual_data(close, cfg, cfg["train_end"], cfg["test_end"])
    with pytest.raises(ValueError): Standardisation.fit([float("nan")])


def test_checkpoint_patience_and_restoration():
    class Weights:
        value = 0
        def get_weights(self): return [np.array([self.value])]
        def set_weights(self, w): self.value = int(w[0][0])
    model, tracker = Weights(), EarlyStopState(2)
    assert not tracker.observe(model, 3, 1)
    model.value = 7
    assert not tracker.observe(model, 2, 2)
    model.value = 9
    assert not tracker.observe(model, 2, 3)
    assert tracker.observe(model, 4, 4)
    tracker.restore(model)
    assert model.value == 7 and tracker.best_epoch == 2
    with pytest.raises(ValueError): tracker.observe(model, np.nan, 5)


def test_selection_band_capacity_variability_and_failures():
    configs = [dict(id="A", units=16), dict(id="B", units=32), dict(id="C", units=16)]
    seeds = [42, 123, 2026]
    runs = []
    for cfg, values in zip(configs, [[1.005, 1.005, 1.005], [1., 1., 1.], [.998, 1.005, 1.012]]):
        for seed, value, epoch in zip(seeds, values, [2, 5, 8]):
            runs.append(dict(configuration_id=cfg["id"], seed=seed, status="complete", best_epoch=epoch,
                             evaluation_metrics=dict(mae_usd=value, rmse_usd=value)))
    result = choose_configuration(runs, configs, seeds)
    assert result["selected_id"] == "A" and result["epochs"] == 5
    runs[0]["status"] = "failed"
    assert choose_configuration(runs, configs, seeds)["selected_id"] == "C"
    with pytest.raises(ValueError): choose_configuration([], configs, seeds)


def test_deterministic_initialisation_shape_and_reload(tmp_path):
    settings = dict(units=16, dropout=.2, learning_rate=.001)
    X = np.linspace(-2., 2., 120, dtype="float32").reshape(4, 30, 1)
    first = build_model(settings, 42)
    expected = predict_standardised(first, X)
    first.save(tmp_path / "untrained.keras")
    import tensorflow as tf
    loaded = tf.keras.models.load_model(tmp_path / "untrained.keras", compile=False)
    np.testing.assert_array_equal(expected, predict_standardised(loaded, X))
    second = build_model(settings, 42)
    np.testing.assert_array_equal(expected, predict_standardised(second, X))
    assert second.count_params() == 1169 and expected.shape == (4,)


def test_training_loop_with_fake_updates_preserves_order_and_best_weights(tmp_path):
    class Counter:
        value = 0
        def numpy(self): return self.value
    class Optimizer:
        iterations = Counter()
    class FakeModel:
        optimizer = Optimizer()
        value = 0.
        seen = []
        def reset_metrics(self): pass
        def train_on_batch(self, x, y):
            self.seen.extend(x[:, 0, 0].tolist())
            self.value += 1
            self.optimizer.iterations.value += 1
            return 1.
        def __call__(self, x, training=False): return np.full((len(x), 1), self.value)
        def get_weights(self): return [np.array([self.value])]
        def set_weights(self, w): self.value = float(w[0][0])
        def save(self, path): path.write_text("synthetic checkpoint")
        def count_params(self): return 1
    model = FakeModel()
    X = np.array([1., 2., 3.]).reshape(3, 1, 1)
    result = train_model(model, X, np.zeros(3), dict(batch_size=2, epochs=8, patience=2),
                         tmp_path, (X, np.full(3, 2.)))
    assert result["best_epoch"] == 1 and result["epochs_completed"] == 3
    assert model.value == 2 and model.seen == [1., 2., 3.] * 3
    history = json.loads((tmp_path / "history.json").read_text())
    assert history[0]["validation_mse"] == 0 and history[-1]["validation_mse"] == 16


def test_existing_run_is_not_overwritten(tmp_path):
    """Reject an occupied run directory before model creation or training."""
    import runpy
    from pathlib import Path
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_lstm_experiments.py"))
    marker = tmp_path / "existing.json"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner["execute_run"](None, None, None, None, None, None, tmp_path)
    assert marker.read_text() == "preserve"



def stage_runner():
    """Import the LSTM runner without starting an experiment phase."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/run_lstm_experiments.py"
    spec = importlib.util.spec_from_file_location("lstm_stage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_repository(tmp_path, *, missing_session=False, future_marker=False):
    """Create synthetic quotes and acquisition metadata entirely in a temporary directory."""
    import hashlib
    root = tmp_path / "synthetic_stage"
    (root / "data/raw").mkdir(parents=True)
    (root / "docs").mkdir()
    config = dict(target_ticker="NVDA", start="2020-01-02", train_end="2020-01-06",
                  validation_end="2020-01-10", test_end="2020-01-17")
    dates = pd.bdate_range("2020-01-02", "2020-01-13")
    prices = list(np.arange(11., 11. + len(dates)))
    quote = {"open": [v - .5 for v in prices], "high": [v + 1 for v in prices],
             "low": [v - 1 for v in prices], "close": prices, "volume": [100] * len(dates)}
    adjusted = [v - .25 for v in prices]
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
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    acquisition = {"retrieved_at_utc": "2020-02-01T00:00:00+00:00", "series": {
        "NVDA": {"sha256": digest}}}
    (root / "docs/data_verification.json").write_text(json.dumps(acquisition))
    return root, config, digest


@pytest.mark.parametrize("end", ["2020-01-01", "2020-01-11", "2020-01-17"])
def test_stage_refuses_outside_development_before_io(tmp_path, monkeypatch, end):
    runner = stage_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path / "no_files_exist")
    def forbidden(*args, **kwargs): raise AssertionError("Bounds must be checked before source access")
    monkeypatch.setattr(runner, "snapshot_provenance", forbidden)
    monkeypatch.setattr(runner, "load_yahoo_snapshot", forbidden)
    config = dict(start="2020-01-02", validation_end="2020-01-10", target_ticker="NVDA")
    with pytest.raises(ValueError, match="development interval"):
        runner.load_stage(config, end)


def test_stage_bounds_real_loader_and_preserves_series_contract(tmp_path, monkeypatch):
    import market_forecasting.data as data_module
    root, config, digest = stage_repository(tmp_path, future_marker=True)
    runner = stage_runner()
    monkeypatch.setattr(runner, "ROOT", root)
    loader, decoder = runner.load_yahoo_snapshot, data_module._number
    bounds, numeric_fields = [], []
    def bounded_loader(path, *, start, end):
        bounds.append((start, end))
        return loader(path, start=start, end=end)
    def selected_number(token, key):
        assert b"UNREAD_FUTURE_QUOTE" not in token
        numeric_fields.append(key)
        return decoder(token, key)
    monkeypatch.setattr(runner, "load_yahoo_snapshot", bounded_loader)
    monkeypatch.setattr(data_module, "_number", selected_number)
    close, actual_digest = runner.load_stage(config, config["validation_end"])
    expected = pd.Series(np.arange(11., 18.),
        index=pd.bdate_range("2020-01-02", "2020-01-10").as_unit("s"), dtype=float)
    pd.testing.assert_series_equal(close, expected, check_freq=False)
    assert actual_digest == digest
    assert bounds == [("2020-01-02", "2020-01-10")]
    assert sum(key != "timestamp" for key in numeric_fields) == 7 * 6
    assert close.name is None and close.index.name is None


def test_stage_refuses_changed_snapshot_before_quote_loading(tmp_path, monkeypatch):
    root, config, _ = stage_repository(tmp_path)
    snapshot = root / "data/raw/NVDA_yahoo_snapshot.json"
    snapshot.write_text(snapshot.read_text() + " ")
    runner = stage_runner()
    monkeypatch.setattr(runner, "ROOT", root)
    def forbidden(*args, **kwargs): raise AssertionError("Identity must be checked before quote loading")
    monkeypatch.setattr(runner, "load_yahoo_snapshot", forbidden)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        runner.load_stage(config, config["validation_end"])


def test_stage_refuses_missing_session(tmp_path, monkeypatch):
    root, config, _ = stage_repository(tmp_path, missing_session=True)
    runner = stage_runner()
    monkeypatch.setattr(runner, "ROOT", root)
    with pytest.raises(ValueError, match="calendar coverage"):
        runner.load_stage(config, config["validation_end"])


@pytest.mark.parametrize("fault", ["unexpected", "duplicate_dates", "ordered"])
def test_stage_refuses_other_calendar_defects(tmp_path, monkeypatch, fault):
    root, config, _ = stage_repository(tmp_path)
    runner = stage_runner()
    monkeypatch.setattr(runner, "ROOT", root)
    audit = {"missing": [], "unexpected": [], "duplicate_dates": 0, "ordered": True}
    audit[fault] = {"unexpected": ["2020-01-04"], "duplicate_dates": 1, "ordered": False}[fault]
    monkeypatch.setattr(runner, "audit_sessions", lambda *args, **kwargs: audit)
    with pytest.raises(ValueError, match="calendar coverage"):
        runner.load_stage(config, config["validation_end"])
