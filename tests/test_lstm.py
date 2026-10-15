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
