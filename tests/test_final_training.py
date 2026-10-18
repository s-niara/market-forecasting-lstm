"""Final-fit and frozen origin-window checks using synthetic exchange sessions."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from market_forecasting.features import (VARIANTS, calculate_features, prepare_feature_dataset,
    prepare_final_feature_dataset, prepare_origin_input, validate_feature_preprocessing)


def final_feature_sample():
    dates = mcal.get_calendar("NASDAQ").schedule("2019-01-01", "2020-12-31").index[:220]
    position = np.arange(len(dates))
    series = {ticker: pd.Series(30 + position * .1 + np.sin(position / (3 + k)), index=dates)
              for k, ticker in enumerate(("NVDA", "SPY", "XLK"))}
    config = dict(start="2019-01-01", train_end=str(dates[159].date()),
                  validation_end=str(dates[199].date()), test_end="2025-12-31",
                  initial_sequence_length=30, price_convention="split_adjusted_close",
                  forecast_horizon_trading_days=1)
    return series, config


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_final_fit_reuses_development_windows_and_scaling(variant):
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    development = prepare_feature_dataset(features, config, variant)
    final = prepare_final_feature_dataset(features, config, variant, config["train_end"])
    for name in ("X", "y", "origins", "targets"):
        np.testing.assert_array_equal(getattr(final, name), getattr(development, name)[development.fit_mask])
    pd.testing.assert_frame_equal(final.metadata, development.metadata.loc[development.fit_mask].reset_index(drop=True))
    expected = dict(development.provenance, validation_examples=0)
    assert final.provenance == expected
    assert final.fit_mask.all()
    validate_feature_preprocessing(final.provenance, variant)


def test_final_fit_excludes_last_label_from_unique_input_scalers():
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    data = prepare_final_feature_dataset(features, config, "C", config["validation_end"])
    assert data.X.shape == (121, 30, 10)
    assert data.provenance["input_unique_rows"] == 150
    assert data.provenance["last_fitting_input_date"] == str(features.index[-2].date())
    assert data.provenance["last_fitting_target_date"] == str(features.index[-1].date())
    for name, saved in data.provenance["input_scalers"].items():
        unique = features[name].iloc[49:-1].to_numpy()
        assert saved["mean"] == pytest.approx(unique.mean())
        assert saved["scale"] == pytest.approx(unique.std(ddof=0))
    np.testing.assert_array_equal(data.targets, features.close.to_numpy()[79:])
    np.testing.assert_array_equal(data.origins, features.close.to_numpy()[78:-1])


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_origin_transform_reproduces_windows_without_target_argument(variant):
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    data = prepare_final_feature_dataset(features, config, variant, config["validation_end"])
    saved = deepcopy(data.provenance)
    for position in (0, 40, len(data.X)-1):
        origin = data.metadata.origin_date.iloc[position]
        prefix = {ticker: values.loc[:origin] for ticker, values in series.items()}
        X, close, metadata = prepare_origin_input(prefix, config, variant, origin, saved)
        np.testing.assert_array_equal(X[0], data.X[position])
        assert close == data.origins[position]
        assert metadata["window_start"] == str(data.metadata.window_start.iloc[position].date())
        assert metadata["origin_date"] == str(origin.date())
        assert X.shape == (1, 30, len(VARIANTS[variant]))
    assert saved == data.provenance


def test_origin_transform_is_causal_and_preserves_recursive_feature_history():
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    data = prepare_final_feature_dataset(features, config, "C", config["validation_end"])
    origin = features.index[149]
    expected = prepare_origin_input(series, config, "C", origin, data.provenance)
    changed = {ticker: values.astype(object).copy() for ticker, values in series.items()}
    # An opaque object raises if converted; all values after the origin must remain unread.
    class UnavailableValue:
        def __float__(self):
            raise AssertionError("A future quote was numerically accessed")
    for values in changed.values():
        values.loc[values.index > origin] = UnavailableValue()
    actual = prepare_origin_input(changed, config, "C", origin, data.provenance)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    with pytest.raises(ValueError, match="session dates"):
        prepare_origin_input({ticker: values.iloc[-80:] for ticker, values in series.items()},
                             config, "C", features.index[-1], data.provenance)


def test_final_interfaces_do_not_process_reserved_rows():
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    expected = prepare_final_feature_dataset(features, config, "A", config["validation_end"])
    class ReservedValue:
        def __float__(self):
            raise AssertionError("Reserved numerical value accessed")
    forbidden = pd.DataFrame([[ReservedValue()] * len(features.columns)],
                             index=pd.DatetimeIndex(["2025-01-02"]), columns=features.columns)
    extended = pd.concat([features, forbidden])
    actual = prepare_final_feature_dataset(extended, config, "A", config["validation_end"])
    np.testing.assert_array_equal(actual.X, expected.X)
    assert actual.provenance == expected.provenance
    tampered = dict(config, validation_end="2025-12-31", test_end="2026-12-31")
    with pytest.raises(ValueError, match="pre-test"):
        prepare_final_feature_dataset(extended, tampered, "A", "2025-01-02")
    with pytest.raises(ValueError, match="pre-test"):
        prepare_origin_input(series, tampered, "A", "2025-01-02", expected.provenance)


@pytest.mark.parametrize("kind", ["order", "scale", "target_scale", "counts", "dates", "names", "start"])
def test_final_preprocessing_metadata_mismatches_stop_origin_inference(kind):
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    data = prepare_final_feature_dataset(features, config, "C", config["validation_end"])
    saved = deepcopy(data.provenance)
    if kind == "order": saved["feature_names"] = saved["feature_names"][::-1]
    elif kind == "scale": saved["input_scalers"]["close"]["scale"] = 0
    elif kind == "target_scale": saved["correction_scaler"]["mean"] = np.nan
    elif kind == "counts": saved["input_unique_rows"] += 1
    elif kind == "dates": saved["last_fitting_input_date"] = saved["last_fitting_target_date"]
    elif kind == "names": saved["input_scalers"].pop("spy_log_return")
    elif kind == "start": saved["first_feature_date"] = "2018-12-31"
    with pytest.raises(ValueError, match="[Ff]rozen|Historical"):
        prepare_origin_input(series, config, "C", features.index[149], saved)


def test_final_origin_rejects_calendar_gaps_non_sessions_and_missing_features():
    series, config = final_feature_sample()
    _, features = calculate_features(series, config)
    data = prepare_final_feature_dataset(features, config, "C", config["validation_end"])
    missing = {ticker: values.drop(features.index[100]) for ticker, values in series.items()}
    with pytest.raises(ValueError, match="session dates"):
        prepare_origin_input(missing, config, "C", features.index[149], data.provenance)
    with pytest.raises(ValueError, match="session dates"):
        prepare_final_feature_dataset(features.drop(features.index[100]), config, "C", config["validation_end"])
    sunday = pd.Timestamp("2019-08-11")
    with pytest.raises(ValueError, match="observed trading session"):
        prepare_origin_input(series, config, "C", sunday, data.provenance)
    bad = {ticker: values.copy() for ticker, values in series.items()}
    bad["SPY"].iloc[100] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        prepare_origin_input(bad, config, "C", features.index[149], data.provenance)
    with pytest.raises(ValueError, match="lookback"):
        prepare_origin_input(series, dict(config, initial_sequence_length=31),
                             "C", features.index[149], data.provenance)


"""Frozen-model verification using synthetic pre-test observations only."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from market_forecasting import arima
from market_forecasting.lstm import build_model, frozen_forecast, load_frozen_model
from market_forecasting.preprocessing import Standardisation


class _SyntheticState:
    def __init__(self, last=10., params=None):
        self.last = last
        self.params = np.array([.3, .4]) if params is None else params.copy()
        self.calls = []

    def forecast(self, steps):
        self.calls.append(("forecast", steps))
        return np.array([self.last + .25])

    def extend(self, observed):
        self.calls.append(("observe", observed[0]))
        return _SyntheticState(observed[0], self.params)


def test_fixed_order_uses_exact_specification_and_records_warnings(monkeypatch):
    import warnings
    from types import SimpleNamespace
    calls = []
    result = SimpleNamespace(params=np.array([.1, .2]), aic=12.,
        param_names=["ar.L1", "sigma2"], mle_retvals={"converged": True},
        filter_results=SimpleNamespace(filtered_state=np.zeros((2, 60))))
    class Model:
        def __init__(self, values, **kwargs): calls.append((values.copy(), kwargs))
        def fit(self, **kwargs):
            calls.append(kwargs)
            warnings.warn("recorded diagnostic", UserWarning)
            return result
    monkeypatch.setattr(arima, "ARIMA", Model)
    fitted, evidence = arima.fit_fixed_order(np.arange(60.) + 10.)
    assert fitted is result and len(calls) == 2
    assert calls[0][1] == dict(order=(2, 1, 2), trend="n", enforce_stationarity=True,
                              enforce_invertibility=True)
    assert calls[1] == {"method_kwargs": {"maxiter": 500}}
    assert evidence["warnings"] == ["recorded diagnostic"]
    assert evidence["converged"] and evidence["finite"] and evidence["training_rows"] == 60
    result.mle_retvals["converged"] = False
    with pytest.raises(RuntimeError, match="failed verification"):
        arima.fit_fixed_order(np.arange(60.) + 10.)
    result.mle_retvals["converged"] = True
    result.filter_results.filtered_state[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="failed verification"):
        arima.fit_fixed_order(np.arange(60.) + 10.)


def test_frozen_arima_requires_forecast_then_matching_observation():
    state = _SyntheticState()
    rolling = arima.FrozenArima(state, "2020-01-02")
    with pytest.raises(ValueError, match="outstanding"):
        rolling.observe("2020-01-03", 11.)
    assert rolling.forecast("2020-01-02", "2020-01-03") == 10.25
    assert state.calls == [("forecast", 1)]
    with pytest.raises(ValueError, match="pending"):
        rolling.forecast("2020-01-02", "2020-01-03")
    with pytest.raises(ValueError, match="outstanding"):
        rolling.observe("2020-01-06", 11.)
    rolling.observe("2020-01-03", 11.)
    assert state.calls == [("forecast", 1), ("observe", 11.)]
    np.testing.assert_array_equal(rolling.state.params, [.3, .4])
    assert rolling.forecast("2020-01-03", "2020-01-06") == 11.25
    rolling.observe("2020-01-06", 12.)
    assert rolling.last_observed_date == pd.Timestamp("2020-01-06")


@pytest.mark.parametrize("origin,target", [("2020-01-02", "2020-01-06"),
    ("2020-01-02", "2020-01-04"), ("2020-01-03", "2020-01-06"),
    ("2020-01-02 09:30", "2020-01-03"), ("2020-01-02", "2025-01-02")])
def test_frozen_arima_rejects_invalid_dates_before_prediction(origin, target):
    state = _SyntheticState()
    rolling = arima.FrozenArima(state, "2020-01-02")
    with pytest.raises(ValueError): rolling.forecast(origin, target)
    assert state.calls == []


def test_frozen_arima_rejects_coefficient_mutation_and_nonfinite_forecast():
    state = _SyntheticState()
    rolling = arima.FrozenArima(state, "2020-01-02")
    state.params[0] = 99.
    with pytest.raises(RuntimeError, match="coefficients"):
        rolling.forecast("2020-01-02", "2020-01-03")
    state.params[0] = .3
    state.last = np.nan
    with pytest.raises(RuntimeError, match="ARIMA forecast"):
        rolling.forecast("2020-01-02", "2020-01-03")
    assert rolling.pending_target is None
    state.last = 10.
    rolling.forecast("2020-01-02", "2020-01-03")
    with pytest.raises(ValueError, match="finite positive"):
        rolling.observe("2020-01-03", np.nan)
    state.extend = lambda observed: _SyntheticState(observed[0], np.array([.4, .5]))
    with pytest.raises(RuntimeError, match="coefficients"):
        rolling.observe("2020-01-03", 11.)
    assert rolling.last_observed_date == pd.Timestamp("2020-01-02")


def test_arima_saved_state_restores_and_continues_synthetic_history(tmp_path):
    from statsmodels.tsa.arima.model import ARIMAResults
    prices = 30 + np.random.default_rng(42).normal(0, .1, 80).cumsum()
    fitted, evidence = arima.fit_fixed_order(prices, order=(1, 1, 0))
    path = tmp_path / "synthetic_arima.pkl"
    fitted.save(path)
    restored = ARIMAResults.load(path)
    np.testing.assert_array_equal(fitted.params, restored.params)
    np.testing.assert_array_equal(fitted.filter_results.filtered_state,
                                  restored.filter_results.filtered_state)
    original = arima.FrozenArima(fitted, "2020-01-02")
    reloaded = arima.FrozenArima(restored, "2020-01-02")
    assert original.forecast("2020-01-02", "2020-01-03") == reloaded.forecast("2020-01-02", "2020-01-03")
    original.observe("2020-01-03", 31.)
    reloaded.observe("2020-01-03", 31.)
    assert original.forecast("2020-01-03", "2020-01-06") == reloaded.forecast("2020-01-03", "2020-01-06")
    np.testing.assert_array_equal(reloaded.state.params, fitted.params)


@pytest.mark.parametrize("features,parameters", [(1, 1169), (8, 1617), (10, 1745)])
def test_frozen_lstm_loads_dimensions_and_preserves_weights_scaler(tmp_path, features, parameters):
    settings = dict(units=16, dropout=.2, learning_rate=.001)
    names = [f"feature_{i}" for i in range(features)]
    model = build_model(settings, 42, features=features)
    X = np.linspace(-1, 1, 2 * 30 * features, dtype="float32").reshape(2, 30, features)
    origins, scaler = np.array([30., 31.]), Standardisation(.05, .3)
    expected = frozen_forecast(model, X, origins, scaler)
    before = model.get_weights()
    path = tmp_path / "untrained.keras"
    model.save(path)
    restored = load_frozen_model(path, names, settings)
    assert restored.count_params() == parameters
    actual = frozen_forecast(restored, X, origins, scaler)
    repeated = frozen_forecast(restored, X, origins, scaler)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual[key], repeated[key])
    np.testing.assert_allclose(actual["prediction"], origins + actual["correction_usd"])
    for a, b in zip(before, restored.get_weights()): np.testing.assert_array_equal(a, b)
    assert scaler == Standardisation(.05, .3)
    with pytest.raises(ValueError, match="dimensions"):
        load_frozen_model(path, names + ["other"], settings)
    with pytest.raises(ValueError, match="unique"):
        load_frozen_model(path, ["close", "close"], settings)
    with pytest.raises(ValueError, match="learning rate"):
        load_frozen_model(path, names, dict(settings, learning_rate=.0003))
    with pytest.raises(ValueError, match="architecture"):
        load_frozen_model(path, names, dict(settings, dropout=0.))


def test_frozen_lstm_rejects_nonfinite_input_scaler_and_model_output():
    model = build_model(dict(units=16, dropout=.2, learning_rate=.001), 42)
    X, origins = np.zeros((1, 30, 1), dtype="float32"), np.array([30.])
    with pytest.raises(ValueError, match="Correction scaler"):
        frozen_forecast(model, X, origins, Standardisation(0., 0.))
    with pytest.raises(ValueError, match="Finite aligned"):
        frozen_forecast(model, np.full_like(X, np.nan), origins, Standardisation(0., 1.))
    with pytest.raises(ValueError, match="Finite aligned"):
        frozen_forecast(model, X[:, :29], origins, Standardisation(0., 1.))
    weights = model.get_weights()
    weights[0][:] = np.nan
    model.set_weights(weights)
    with pytest.raises((ValueError, RuntimeError), match="[Nn]onfinite"):
        frozen_forecast(model, X, origins, Standardisation(0., 1.))


def test_frozen_lstm_rejects_inference_that_mutates_weights():
    class MutatingModel:
        input_shape = (None, 30, 1)
        value = 0.
        def get_weights(self): return [np.array([self.value])]
        def __call__(self, X, training=False):
            assert not training
            self.value += 1
            return np.zeros((len(X), 1))
    with pytest.raises(RuntimeError, match="weights changed"):
        frozen_forecast(MutatingModel(), np.zeros((1, 30, 1)), np.array([10.]),
                        Standardisation(0., 1.))


"""Benchmark-relative accuracy checks using synthetic, aligned USD prices."""
import numpy as np
import pytest
from market_forecasting.dataset import benchmark_relative_metrics


def test_benchmark_relative_metrics_match_independent_arithmetic():
    actual = np.array([10., 20., 30.])
    predicted = actual + np.array([1., -2., 3.])
    naive = actual + np.array([2., -4., 6.])
    result = benchmark_relative_metrics(actual, predicted, naive)
    assert result["model"] == {"n": 3, "mae_usd": 2., "rmse_usd": pytest.approx(np.sqrt(14 / 3))}
    assert result["naive"]["mae_usd"] == 4.
    assert result["naive"]["rmse_usd"] == pytest.approx(np.sqrt(56 / 3))
    assert result["absolute_skill_usd"]["mae"] == 2.
    assert result["absolute_skill_usd"]["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert result["percentage_skill"] == {"mae": 50., "rmse": 50.}
    worse = benchmark_relative_metrics(actual, naive, predicted)
    assert worse["absolute_skill_usd"]["mae"] == -2.
    assert worse["percentage_skill"] == {"mae": -100., "rmse": -100.}


def test_benchmark_relative_metrics_zero_baseline_has_undefined_percentage():
    for predicted in ([10., 20.], [11., 21.]):
        result = benchmark_relative_metrics([10., 20.], predicted, [10., 20.])
        assert result["percentage_skill"] == {"mae": None, "rmse": None}
        assert result["absolute_skill_usd"]["mae"] == -result["model"]["mae_usd"]
        assert result["absolute_skill_usd"]["rmse"] == -result["model"]["rmse_usd"]


@pytest.mark.parametrize("actual,predicted,naive", [
    ([10., 20.], [10.], [10., 20.]),
    ([10., 20.], [10., 20.], [10.]),
    ([[10., 20.]], [[10., 20.]], [[10., 20.]]),
    ([], [], []),
    ([10., np.nan], [10., 20.], [10., 20.]),
    ([10., 20.], [np.inf, 20.], [10., 20.]),
    ([10., 20.], [10., 20.], [10., np.nan]),
])
def test_benchmark_relative_metrics_rejects_misaligned_or_nonfinite_vectors(actual, predicted, naive):
    with pytest.raises(ValueError): benchmark_relative_metrics(actual, predicted, naive)


"""Final-stage orchestration guards using synthetic payloads and mocked model calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest


def final_runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    # A repository runner import never executes fitting or the command-line entry point.
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("final_protocol_test", root / "scripts/run_final_training.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def approved_final_settings():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/final_training_v1.json").read_text())


def forbidden_computation(*args, **kwargs):
    raise AssertionError("Rejected requests must stop before data access or model computation")


FROZEN_PROTOCOL_MUTATIONS = [
    (("data", "start"), "2016-01-01"),
    (("data", "fit_end"), "2025-01-02"),
    (("data", "price_convention"), "adjusted_close"),
    (("data", "bounded_quote_decoding_required"), False),
    (("common_eligibility", "lookback"), 31),
    (("common_eligibility", "warmup_rows"), 48),
    (("common_eligibility", "include_last_target_in_input_scaler"), True),
    (("arima", "order"), [1, 1, 1]),
    (("arima", "trend"), "t"),
    (("arima", "maxiter"), 1000),
    (("lstm", "units"), 32),
    (("lstm", "dropout"), 0.0),
    (("lstm", "learning_rate"), 0.0003),
    (("lstm", "batch_size"), 64),
    (("lstm", "epochs"), 20),
    (("lstm", "seeds"), [42, 123, 2027]),
    (("lstm", "early_stopping"), True),
    (("lstm", "internal_validation"), True),
    (("lstm", "formulation"), "return_correction"),
    (("lstm", "feature_names", "A"), ["close", "spy_log_return"]),
    (("lstm", "feature_specification", "rsi"), 10),
    (("comparison", "primary_lstm"), "C"),
    (("comparison", "prediction_ensemble"), True),
    (("evaluation_protocol", "test_access_authorised"), True),
    (("evaluation_protocol", "execution_authorised"), True),
    (("evaluation_protocol", "test_sealed"), False),
]


def established_settings(root):
    return [json.loads((root / 'configs' / name).read_text()) for name in
            ('nvda.json', 'lstm_selected_v1.json', 'feature_experiment_v1.json', 'arima.json')]


def test_approved_final_protocol_passes_without_data_access(monkeypatch):
    runner = final_runner(monkeypatch)
    monkeypatch.setattr(runner, 'load_stage', forbidden_computation)
    runner.guard_protocol(approved_final_settings(), *established_settings(runner.ROOT))


@pytest.mark.parametrize('path,value', FROZEN_PROTOCOL_MUTATIONS)
def test_changed_final_protocol_rejected_before_loading(monkeypatch, path, value):
    runner = final_runner(monkeypatch)
    protocol = approved_final_settings()
    parent = protocol
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    monkeypatch.setattr(runner, 'load_stage', forbidden_computation)
    with pytest.raises(ValueError):
        runner.guard_protocol(protocol, *established_settings(runner.ROOT))


def test_sealed_forecast_rejected_before_feature_or_model_calls(monkeypatch):
    runner = final_runner(monkeypatch)
    monkeypatch.setattr(runner, 'prepare_origin_input', forbidden_computation)
    monkeypatch.setattr(runner, 'frozen_forecast', forbidden_computation)
    with pytest.raises(ValueError, match='sealed'):
        runner.pretest_forecast({}, {}, '2024-12-31', '2025-01-02', {}, None)


def test_nonconsecutive_forecast_dates_rejected(monkeypatch):
    runner = final_runner(monkeypatch)
    monkeypatch.setattr(runner, 'prepare_origin_input', forbidden_computation)
    with pytest.raises(ValueError, match='consecutive'):
        runner.pretest_forecast({}, {}, '2024-12-26', '2024-12-30', {}, None)


def test_forecast_alignment_and_target_free_contract(monkeypatch):
    runner = final_runner(monkeypatch)
    calls = []
    def prepare(series, config, variant, origin, preprocessing):
        calls.append(('features', str(origin.date()), variant))
        return np.ones((1, 30, len(runner.VARIANTS[variant]))), 20., {}
    def forecast(model, X, origins, scaler):
        assert origins.tolist() == [20.]
        return {'prediction': np.array([20.5])}
    class Cursor:
        def forecast(self, origin, target):
            calls.append(('arima_forecast', str(target.date())))
            return 20.25
        def observe(self, *args):
            raise AssertionError('Issuing forecasts must not observe the target')
    monkeypatch.setattr(runner, 'prepare_origin_input', prepare)
    monkeypatch.setattr(runner, 'frozen_forecast', forecast)
    bundles = {f'{v}_seed_{seed}': {'variant': v, 'preprocessing': {}, 'model': None,
               'correction_scaler': None} for v in runner.VARIANTS for seed in [42, 123, 2026]}
    result = runner.pretest_forecast({}, {}, '2024-12-27', '2024-12-30', bundles, Cursor())
    assert result['origin_date'] == '2024-12-27' and result['target_date'] == '2024-12-30'
    assert len(result['predictions']) == 11 and result['predictions']['naive'] == 20.
    assert len([c for c in calls if c[0] == 'features']) == 9


def test_hash_mismatch_stops_before_artifact_loading(monkeypatch, tmp_path):
    runner = final_runner(monkeypatch)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    artifact = tmp_path / 'model.keras'
    artifact.write_bytes(b'synthetic artifact')
    hashes = {'model.keras': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    runner.verify_hashes(hashes)
    artifact.write_bytes(b'altered artifact')
    with pytest.raises(ValueError, match='Integrity mismatch'):
        runner.verify_hashes(hashes)


def test_evaluation_command_stops_before_runtime_or_source_access(monkeypatch):
    runner = final_runner(monkeypatch)
    monkeypatch.setattr(runner, 'INDEPENDENT_EVALUATION_AUTHORISED', False)
    monkeypatch.setattr(runner, 'configure_runtime', forbidden_computation)
    monkeypatch.setattr(runner, 'approved_configuration', forbidden_computation)
    monkeypatch.setattr(sys, 'argv', ['run_final_training.py', 'evaluate'])
    with pytest.raises(SystemExit, match='sealed'):
        runner.main()


def test_existing_final_directory_refuses_loading_or_refitting(monkeypatch, tmp_path):
    runner = final_runner(monkeypatch)
    protocol = approved_final_settings()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'approved_configuration', lambda: (protocol, {}))
    monkeypatch.setattr(runner, 'prepare_final', forbidden_computation)
    (tmp_path / protocol['artifacts']['directory']).mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        runner.fit_and_freeze()


def test_failed_arima_stops_before_lstm_fits_and_preserves_failure(monkeypatch, tmp_path):
    runner = final_runner(monkeypatch)
    protocol = approved_final_settings()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path / 'artifacts').mkdir()
    series = {'NVDA': pd.Series([10., 11.], index=pd.to_datetime(['2024-12-30','2024-12-31']))}
    monkeypatch.setattr(runner, 'approved_configuration', lambda: (protocol, {}))
    monkeypatch.setattr(runner, 'prepare_final', lambda *args: (series, {}, {}))
    monkeypatch.setattr(runner, 'protected_hashes', lambda: {})
    monkeypatch.setattr(runner, 'source_snapshot', lambda output: {})
    monkeypatch.setattr(runner, 'build_model', forbidden_computation)
    def failed_fit(*args, **kwargs):
        raise RuntimeError('synthetic nonconverged fit')
    monkeypatch.setattr(runner, 'fit_fixed_order', failed_fit)
    with pytest.raises(RuntimeError, match='nonconverged'):
        runner.fit_and_freeze()
    manifest = json.loads((tmp_path / protocol['artifacts']['directory'] / 'manifest.json').read_text())
    assert manifest['status'] == manifest['arima']['status'] == 'failed'
    assert manifest['runs'] == []
    assert 'nonconverged' in manifest['error']
