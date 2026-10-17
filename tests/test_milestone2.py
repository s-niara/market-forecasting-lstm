"""Synthetic checks for window alignment, date boundaries and benchmark metrics."""
import numpy as np
import pandas as pd
import pytest
from market_forecasting.calendar import audit_sessions
from market_forecasting.dataset import build_development_dataset,naive_predictions,error_metrics


def config():
    """Return short synthetic split boundaries and a two-session lookback."""
    return dict(price_convention='split_adjusted_close',start='2023-01-01',train_end='2023-01-05',validation_end='2023-01-10',test_end='2023-01-20',forecast_horizon_trading_days=1,initial_sequence_length=2)


def test_window_alignment_and_boundary_context():
    """Verify input/target offsets and legitimate context across a split boundary."""
    dates=pd.bdate_range('2023-01-02','2023-01-13')
    values=pd.Series(np.arange(1,len(dates)+1,dtype=float),index=dates)
    ds=build_development_dataset(values,config())
    np.testing.assert_array_equal(ds.X[0,:,0],[1,2])
    assert ds.y[0]==3
    assert (ds.metadata.origin_date < ds.metadata.target_date).all()
    first=ds.metadata[ds.metadata.split=='validation'].iloc[0]
    assert first.origin_date==pd.Timestamp('2023-01-05')
    assert first.target_date==pd.Timestamp('2023-01-06')
    np.testing.assert_array_equal(naive_predictions(ds),ds.X[:,-1,0])


def test_test_values_cannot_change_development_windows():
    """Check that post-validation price changes cannot affect development arrays."""
    dates=pd.bdate_range('2023-01-02','2023-01-13')
    values=pd.Series(np.arange(1,len(dates)+1,dtype=float),index=dates)
    a=build_development_dataset(values,config())
    values.loc['2023-01-11':]=np.nan
    b=build_development_dataset(values,config())
    np.testing.assert_array_equal(a.X,b.X);np.testing.assert_array_equal(a.y,b.y)
    assert b.metadata.target_date.max()<=pd.Timestamp(config()['validation_end'])


def test_missing_session_and_unexpected_day():
    """Detect a missing session and an incorrectly included market holiday."""
    audit=audit_sessions(pd.DatetimeIndex(['2024-07-03','2024-07-04']),'2024-07-03','2024-07-05','NASDAQ')
    assert audit['missing']==['2024-07-05']
    assert audit['unexpected']==['2024-07-04']


def test_special_closure():
    """Verify the calendar excludes the January 2025 special market closure."""
    audit=audit_sessions(pd.DatetimeIndex(['2025-01-08','2025-01-10']),'2025-01-08','2025-01-10','NASDAQ')
    assert audit['missing']==[] and audit['unexpected']==[]


def test_metrics_known_values():
    """Compare errors with hand-computed values and reject nonfinite predictions."""
    m=error_metrics([1,3],[2,1]);assert m['mae_usd']==1.5
    assert m['rmse_usd']==pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError):error_metrics([1],[float('nan')])


def test_unapproved_convention_rejected():
    """Reject a target convention that is not the implemented split-adjusted Close."""
    c=config();c['price_convention']='unresolved'
    with pytest.raises(ValueError,match='explicitly agreed'):
        build_development_dataset(pd.Series([1.,2.,3.],index=pd.bdate_range('2023-01-02',periods=3)),c)


@pytest.mark.parametrize("columns", [1, 2])
def test_dataframe_input_cannot_create_extra_feature_axes(columns):
    """Require the documented one-dimensional Close Series contract."""
    dates = pd.bdate_range("2020-01-02", periods=7)
    values = pd.DataFrame(np.ones((len(dates), columns)), index=dates)
    with pytest.raises((TypeError, ValueError), match="Series"):
        build_development_dataset(values, window_config())


def window_config():
    return dict(price_convention="split_adjusted_close", start="2020-01-02",
                train_end="2020-01-06", validation_end="2020-01-10", test_end="2020-01-17",
                forecast_horizon_trading_days=1, initial_sequence_length=2)


def window_prices():
    return pd.Series([11., 12., 13., 14., 15., 16., 17.],
                     index=pd.bdate_range("2020-01-02", "2020-01-10"), name="close")


def test_explicit_first_transition_and_last_windows():
    dataset = build_development_dataset(window_prices(), window_config())
    assert dataset.X.shape == (5, 2, 1)
    assert dataset.y.shape == (5,)
    # Explicit examples detect offset errors independently of the construction loop.
    expected = [
        (0, [11., 12.], 13., "2020-01-02", "2020-01-03", "2020-01-06", "train"),
        (1, [12., 13.], 14., "2020-01-03", "2020-01-06", "2020-01-07", "validation"),
        (4, [15., 16.], 17., "2020-01-08", "2020-01-09", "2020-01-10", "validation"),
    ]
    for i, inputs, target, start, origin, date, split in expected:
        np.testing.assert_array_equal(dataset.X[i, :, 0], inputs)
        assert dataset.y[i] == target
        row = dataset.metadata.iloc[i]
        assert row.window_start == pd.Timestamp(start)
        assert row.origin_date == pd.Timestamp(origin)
        assert row.target_date == pd.Timestamp(date)
        assert row.split == split
    assert dataset.metadata.split.value_counts().to_dict() == {"validation": 4, "train": 1}


def test_target_change_cannot_enter_its_own_forecast():
    original = window_prices()
    before = build_development_dataset(original, window_config())
    changed = original.copy()
    changed.loc["2020-01-08"] += 50
    after = build_development_dataset(changed, window_config())
    # The perturbed target is unavailable to its own forecast and all earlier ones.
    np.testing.assert_array_equal(after.X[:3], before.X[:3])
    np.testing.assert_array_equal(naive_predictions(after)[:3], naive_predictions(before)[:3])
    np.testing.assert_array_equal(after.y[:2], before.y[:2])
    assert after.y[2] == before.y[2] + 50
    # It becomes legitimate observed context for the following session's forecast.
    assert naive_predictions(after)[3] == naive_predictions(before)[3] + 50
    pd.testing.assert_frame_equal(after.metadata, before.metadata)


def test_naive_predictions_are_an_independent_copy():
    dataset = build_development_dataset(window_prices(), window_config())
    forecasts = naive_predictions(dataset)
    forecasts[0] = -100
    assert dataset.X[0, -1, 0] == 12


@pytest.mark.parametrize("actual,predicted", [
    ([], []), ([1., 2.], [1.]), ([[1.]], [[1.]]), ([1.], [[1.]]), (1., 1.),
    ([np.nan], [1.]), ([np.inf], [1.]), ([1.], [np.nan]), ([1.], [np.inf]),
])
def test_metrics_reject_invalid_shapes_or_nonfinite_inputs(actual, predicted):
    with pytest.raises(ValueError):
        error_metrics(actual, predicted)


def load_baseline_script():
    """Import the executable without running its experiment entry point."""
    import importlib.util
    from pathlib import Path
    import market_forecasting.dataset as dataset_module
    root = Path(dataset_module.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("notebook02_baseline_test", root / "scripts/build_naive_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_repository(tmp_path, *, missing_session=False, future_marker=False):
    """Create only synthetic quotes and their matching acquisition checksum."""
    import hashlib
    import json
    root = tmp_path / "synthetic_repository"
    for folder in ("configs", "data/raw", "docs", "artifacts"):
        (root / folder).mkdir(parents=True)
    settings = dict(window_config(), target_ticker="NVDA")
    (root / "configs/nvda.json").write_text(json.dumps(settings))
    dates = pd.bdate_range("2020-01-02", "2020-01-13")
    prices = list(np.arange(11., 11. + len(dates)))
    quote = {"open": [v - .5 for v in prices], "high": [v + 1 for v in prices],
             "low": [v - 1 for v in prices], "close": prices,
             "volume": list(range(100, 100 + len(dates)))}
    adjusted = [v - .25 for v in prices]
    timestamps = [int((d.tz_localize("America/New_York") + pd.Timedelta(hours=16)).timestamp())
                  for d in dates]
    if missing_session:
        location = dates.get_loc(pd.Timestamp("2020-01-07"))
        timestamps.pop(location)
        for values in quote.values():
            values.pop(location)
        adjusted.pop(location)
    if future_marker:
        quote["close"][-1] = "UNREAD_FUTURE_QUOTE"
    payload = {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"}, "timestamp": timestamps,
        "indicators": {"quote": [quote], "adjclose": [{"adjclose": adjusted}]}}]}}
    source = root / "data/raw/NVDA_yahoo_snapshot.json"
    source.write_text(json.dumps(payload))
    acquisition = {"retrieved_at_utc": "2020-02-01T00:00:00+00:00", "series": {
        "NVDA": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}
    (root / "docs/data_verification.json").write_text(json.dumps(acquisition))
    return root


def test_baseline_script_records_correct_synthetic_result(tmp_path):
    import json
    root = baseline_repository(tmp_path)
    script = load_baseline_script()
    result = script.main(root)
    assert result["input_shape"] == [5, 2, 1]
    assert result["train_examples"] == 1
    assert result["validation_examples"] == 4
    assert result["validation_metrics"] == {"n": 4, "mae_usd": 1., "rmse_usd": 1.}
    assert result["first_validation_origin"] == "2020-01-06"
    assert result["first_validation_target"] == "2020-01-07"
    assert result["last_target"] == "2020-01-10"
    assert result["test_evaluated"] is False
    output = root / "artifacts/naive_v1"
    assert json.loads((output / "summary.json").read_text()) == result
    table = pd.read_json(output / "validation_predictions.json")
    np.testing.assert_array_equal(table.actual_close, [14., 15., 16., 17.])
    np.testing.assert_array_equal(table.prediction, [13., 14., 15., 16.])
    arrays = np.load(output / "development_windows.npz")
    np.testing.assert_array_equal(arrays["X"][:, -1, 0], [12., 13., 14., 15., 16.])
    np.testing.assert_array_equal(arrays["y"], [13., 14., 15., 16., 17.])


def test_baseline_script_bounds_loading_before_decoding_future_quotes(tmp_path, monkeypatch):
    import market_forecasting.data as data_module
    root = baseline_repository(tmp_path, future_marker=True)
    script = load_baseline_script()
    actual_loader = script.load_yahoo_snapshot
    actual_number = data_module._number
    calls = []
    def bounded_loader(path, *, start, end):
        calls.append((start, end))
        return actual_loader(path, start=start, end=end)
    def selected_number(token, key):
        assert b"UNREAD_FUTURE_QUOTE" not in token
        return actual_number(token, key)
    monkeypatch.setattr(script, "load_yahoo_snapshot", bounded_loader)
    monkeypatch.setattr(data_module, "_number", selected_number)
    result = script.main(root)
    assert calls == [("2020-01-02", "2020-01-10")]
    assert result["validation_metrics"]["mae_usd"] == 1.


def test_baseline_script_rejects_missing_session_before_writing(tmp_path):
    root = baseline_repository(tmp_path, missing_session=True)
    script = load_baseline_script()
    with pytest.raises(ValueError):
        script.main(root)
    assert not (root / "artifacts/naive_v1").exists()


@pytest.mark.parametrize("fault", ["unexpected", "duplicate_dates", "ordered"])
def test_baseline_script_rejects_all_calendar_defects(tmp_path, monkeypatch, fault):
    root = baseline_repository(tmp_path)
    script = load_baseline_script()
    audit = {"missing": [], "unexpected": [], "duplicate_dates": 0, "ordered": True}
    audit[fault] = {"unexpected": ["2020-01-04"], "duplicate_dates": 1, "ordered": False}[fault]
    monkeypatch.setattr(script, "audit_sessions", lambda *args, **kwargs: audit)
    with pytest.raises(ValueError):
        script.main(root)
    assert not (root / "artifacts/naive_v1").exists()


def test_baseline_script_rejects_checksum_mismatch_before_loading(tmp_path, monkeypatch):
    root = baseline_repository(tmp_path)
    source = root / "data/raw/NVDA_yahoo_snapshot.json"
    source.write_text(source.read_text() + " ")
    script = load_baseline_script()
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Identity must be verified before loading records")
    monkeypatch.setattr(script, "load_yahoo_snapshot", forbidden_loader)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        script.main(root)
    assert not (root / "artifacts/naive_v1").exists()


def test_baseline_script_refuses_overwrite_before_loading(tmp_path, monkeypatch):
    root = baseline_repository(tmp_path)
    output = root / "artifacts/naive_v1"
    output.mkdir()
    marker = output / "preserved.txt"
    marker.write_text("existing experiment")
    script = load_baseline_script()
    def forbidden(*args, **kwargs):
        raise AssertionError("An existing run must be rejected before any source access")
    monkeypatch.setattr(script, "load_yahoo_snapshot", forbidden)
    monkeypatch.setattr(script, "snapshot_provenance", forbidden)
    with pytest.raises(FileExistsError):
        script.main(root)
    assert marker.read_text() == "existing experiment"
    assert list(output.iterdir()) == [marker]
