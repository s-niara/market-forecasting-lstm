"""Independent indicator checks and temporal safeguards without experimental training."""
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import pytest
from market_forecasting.features import (seeded_ema, wilder_rsi, calculate_features,
                                         prepare_feature_dataset, VARIANTS)
from market_forecasting.preprocessing import reconstruct_prices


def sample():
    """Create aligned synthetic sessions with a reserved future block."""
    dates = pd.bdate_range("2020-01-01", periods=220)
    x = np.arange(len(dates))
    series = {ticker: pd.Series(30 + x * .1 + np.sin(x / (3 + k)), index=dates)
              for k, ticker in enumerate(("NVDA", "SPY", "XLK"))}
    config = dict(start=str(dates[0].date()), train_end=str(dates[159].date()),
                  validation_end=str(dates[199].date()), test_end=str(dates[-1].date()),
                  initial_sequence_length=30, price_convention="split_adjusted_close",
                  forecast_horizon_trading_days=1)
    return series, config


def test_ema_arithmetic_seed_and_recursive_values():
    result = seeded_ema([np.nan, 2, 4, 9, 8, 12], 3)
    # First seed is 5; alpha is 1/2, giving 6.5 and then 9.25.
    assert np.isnan(result[:3]).all()
    np.testing.assert_allclose(result[3:], [5, 6.5, 9.25])
    assert np.isnan(seeded_ema([1, 2], 3)).all()
    with pytest.raises(ValueError): seeded_ema([1, np.nan, 3], 2)


def test_rsi_wilder_seed_recursion_and_edge_cases():
    prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
    result = wilder_rsi(prices)
    # First 14 changes total 3.34 gain and 1.40 loss; next change is -0.28.
    expected_seed = 100 * 3.34 / (3.34 + 1.40)
    gain, loss = (3.34 / 14 * 13) / 14, (1.40 / 14 * 13 + .28) / 14
    np.testing.assert_allclose(result[14:], [expected_seed, 100 * gain / (gain + loss)])
    assert np.isnan(result[:14]).all()
    for values, expected in [(np.arange(30), 100), (-np.arange(30), 0), (np.ones(30), 50)]:
        np.testing.assert_array_equal(wilder_rsi(values)[14:], np.full(16, expected))


def test_indicators_first_valid_positions_and_linear_price_identities():
    series, cfg = sample()
    for key in series: series[key] = pd.Series(np.arange(1, 221, dtype=float), index=series[key].index)
    raw, f = calculate_features(series, cfg)
    for column, position in {"sma20": 19, "sma50": 49, "ema12": 11, "ema26": 25,
                             "rsi14": 14, "macd": 25, "signal": 33, "histogram": 33}.items():
        assert raw[column].first_valid_index() == raw.index[position]
    # A linear ramp has EMA(n,t) = Close(t) - (n-1)/2 under arithmetic seeding.
    np.testing.assert_allclose(raw.ema12.iloc[11:], raw.close.iloc[11:] - 5.5)
    np.testing.assert_allclose(raw.macd.iloc[25:], 7)
    np.testing.assert_allclose(raw.signal.iloc[33:], 7)
    np.testing.assert_allclose(raw.histogram.iloc[33:], 0, atol=1e-12)
    assert f.sma50_distance.iloc[49] == pytest.approx(50 / 25.5 - 1)
    assert f.spy_log_return.iloc[1] == pytest.approx(np.log(2))
    np.testing.assert_allclose(f.macd_relative.iloc[25:], 7 / raw.close.iloc[25:])


def test_common_eligibility_alignment_and_unique_training_scalers():
    series, cfg = sample()
    _, f = calculate_features(series, cfg)
    datasets = {v: prepare_feature_dataset(f, cfg, v) for v in VARIANTS}
    for variant, d in datasets.items():
        assert d.X.shape == (121, 30, len(VARIANTS[variant]))
        assert d.fit_mask.sum() == 81
        assert d.provenance["input_unique_rows"] == 110
        assert d.metadata.target_date.iloc[0] == f.index[79]
        assert d.metadata.window_start.iloc[0] == f.index[49]
        assert d.metadata.origin_date.iloc[0] == f.index[78]
        pd.testing.assert_frame_equal(d.metadata, datasets["A"].metadata)
        np.testing.assert_array_equal(d.y, datasets["A"].y)
        for name, scaler in d.provenance["input_scalers"].items():
            assert scaler["mean"] == pytest.approx(f[name].iloc[49:159].mean())
            assert scaler["scale"] == pytest.approx(f[name].iloc[49:159].std(ddof=0))
        np.testing.assert_allclose(reconstruct_prices(d.origins, d.y, d.correction_scaler), d.targets, atol=1e-7)
        # The first validation input includes training context and ends before its label.
        assert d.metadata.loc[81, "origin_date"] == f.index[159]


def test_future_perturbation_preserves_earlier_features_scalers_and_predictions():
    series, cfg = sample()
    _, f = calculate_features(series, cfg)
    changed = {k: v.copy() for k, v in series.items()}
    for s in changed.values(): s.iloc[180:] *= 20
    _, altered = calculate_features(changed, cfg)
    pd.testing.assert_frame_equal(f.iloc[:180], altered.iloc[:180])
    # Prefix-only calculation must match uninterrupted state across the fit boundary.
    prefix_cfg = dict(cfg, validation_end=str(f.index[179].date()))
    _, prefix = calculate_features(series, prefix_cfg)
    pd.testing.assert_frame_equal(f.iloc[:180], prefix)
    from market_forecasting.lstm import build_model, predict_standardised
    for variant in VARIANTS:
        a, b = [prepare_feature_dataset(frame, cfg, variant) for frame in (f, altered)]
        assert a.provenance == b.provenance
        mask = (a.metadata.target_date <= f.index[180]).to_numpy()
        np.testing.assert_array_equal(a.X[mask], b.X[mask])
        model = build_model(dict(units=16, dropout=.2, learning_rate=.001), 42, features=a.X.shape[2])
        # Untrained inference is enough to verify that unavailable future prices cannot enter forecasts.
        pa, pb = [reconstruct_prices(d.origins[mask], predict_standardised(model, d.X[mask]), d.correction_scaler) for d in (a, b)]
        np.testing.assert_array_equal(pa, pb)
        assert model.count_params() == {"A": 1169, "B": 1617, "C": 1745}[variant]


def test_test_period_prices_are_excluded_before_feature_computation():
    series, cfg = sample()
    raw, f = calculate_features(series, cfg)
    for s in series.values(): s.iloc[200:] = np.nan
    new_raw, new_f = calculate_features(series, cfg)
    pd.testing.assert_frame_equal(raw, new_raw)
    pd.testing.assert_frame_equal(f, new_f)
    assert len(f) == 200


def test_missing_sessions_and_internal_values_are_rejected():
    series, cfg = sample()
    missing = dict(series, SPY=series["SPY"].drop(series["SPY"].index[100]))
    with pytest.raises(ValueError, match="sessions"): calculate_features(missing, cfg)
    series["XLK"].iloc[100] = np.nan
    with pytest.raises(ValueError, match="Missing"): calculate_features(series, cfg)
    series, cfg = sample()
    _, f = calculate_features(series, cfg)
    f.loc[f.index[100], "rsi14_fraction"] = np.nan
    with pytest.raises(ValueError, match="warm-up"): prepare_feature_dataset(f, cfg, "A")


def test_normalised_indicators_are_price_scale_invariant():
    series, cfg = sample()
    _, f = calculate_features(series, cfg)
    _, g = calculate_features({k: s * 10 for k, s in series.items()}, cfg)
    np.testing.assert_allclose(f.iloc[:, 1:], g.iloc[:, 1:], rtol=1e-10, atol=1e-12, equal_nan=True)


def test_run_overwrite_protection_and_fixed_protocol(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    runner = runpy.run_path(str(root / "scripts/run_feature_experiments.py"))
    with pytest.raises(FileExistsError):
        runner["execute_feature_run"](None, None, None, None, None, tmp_path)
    cfg = json.loads((root / "configs/feature_experiment_v1.json").read_text())
    assert cfg["seeds"] == [42, 123, 2026] and len(cfg["variants"]) == 3
    assert cfg["epochs"] == 19 and not cfg["early_stopping"]



def test_input_channels_follow_declared_features_and_independent_scaling():
    series, config = sample()
    _, features = calculate_features(series, config)
    expected_features = {
        "A": ["close"],
        "B": ["close", "sma20_distance", "sma50_distance", "ema12_distance", "ema26_distance",
              "rsi14_fraction", "macd_relative", "histogram_relative"],
        "C": ["close", "sma20_distance", "sma50_distance", "ema12_distance", "ema26_distance",
              "rsi14_fraction", "macd_relative", "histogram_relative", "spy_log_return", "xlk_log_return"],
    }
    assert VARIANTS == expected_features
    for variant, names in expected_features.items():
        data = prepare_feature_dataset(features, config, variant)
        assert data.provenance["feature_names"] == names
        # Compare first, first-validation and last windows with independent column-wise arithmetic.
        for position in [0, 81, 120]:
            row = data.metadata.iloc[position]
            columns = []
            for name in names:
                fitting = features[name].iloc[49:159].to_numpy()
                observations = features.loc[row.window_start:row.origin_date, name].to_numpy()
                columns.append((observations - np.mean(fitting)) / np.std(fitting, ddof=0))
            expected = np.column_stack(columns).astype("float32")
            assert expected.shape == (30, len(names))
            np.testing.assert_array_equal(data.X[position], expected)
        # Input DataFrame storage order must never change the model's feature-axis meaning.
        reordered = prepare_feature_dataset(features.iloc[:, ::-1], config, variant)
        np.testing.assert_array_equal(data.X, reordered.X)
        assert data.provenance == reordered.provenance


def test_shared_feature_channels_and_targets_are_identical_across_variants():
    series, config = sample()
    _, features = calculate_features(series, config)
    a, b, c = [prepare_feature_dataset(features, config, variant) for variant in ("A", "B", "C")]
    np.testing.assert_array_equal(a.X, b.X[:, :, :1])
    np.testing.assert_array_equal(b.X, c.X[:, :, :len(VARIANTS["B"])])
    for other in (b, c):
        np.testing.assert_array_equal(a.origins, other.origins)
        np.testing.assert_array_equal(a.targets, other.targets)
        np.testing.assert_array_equal(a.y, other.y)
        np.testing.assert_array_equal(a.fit_mask, other.fit_mask)
        assert a.correction_scaler == other.correction_scaler
        pd.testing.assert_frame_equal(a.metadata, other.metadata)


def feature_runner(monkeypatch):
    """Import the command-line runner without executing its entry point."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("feature_protocol_test", root / "scripts/run_feature_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key,altered", [
    ("sma", [10, 50]), ("ema", [10, 26]), ("rsi", 10), ("macd", [12, 26, 8]),
    ("optimizer", "SGD"), ("loss", "mean_absolute_error"), ("formulation", "direct_price"),
])
def test_fixed_feature_specification_is_checked_before_loading_or_training(tmp_path, monkeypatch, key, altered):
    root = tmp_path / "synthetic_protocol"
    (root / "configs").mkdir(parents=True)
    common = dict(units=16, dropout=.2, batch_size=32, learning_rate=.001)
    selected = dict(configuration=dict(id="C", **common), epochs=19, lookback=30,
                    seeds=[42, 123, 2026])
    settings = dict(common, epochs=19, lookback=30, seeds=[42, 123, 2026],
                    variants=VARIANTS, early_stopping=False, warmup_rows=49,
                    sma=[20, 50], ema=[12, 26], rsi=14, macd=[12, 26, 9],
                    optimizer="Adam", loss="standardised_correction_mse",
                    formulation="residual_dollar_correction")
    settings[key] = altered
    (root / "configs/nvda.json").write_text("{}")
    (root / "configs/lstm_selected_v1.json").write_text(json.dumps(selected))
    (root / "configs/feature_experiment_v1.json").write_text(json.dumps(settings))
    runner = feature_runner(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "configure_runtime", lambda: None)
    def forbidden(*args, **kwargs): raise AssertionError("Protocol must be checked before data access or fitting")
    monkeypatch.setattr(runner, "load_stage", forbidden)
    monkeypatch.setattr(runner, "calculate_features", forbidden)
    monkeypatch.setattr(runner, "execute_feature_run", forbidden)
    monkeypatch.setattr(runner, "train_model", forbidden)
    with pytest.raises(ValueError, match=f"protocol mismatch for {key}"):
        runner.main()
    assert not (root / "artifacts").exists()
