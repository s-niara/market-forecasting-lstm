"""Synthetic snapshot tests for integrity, exchange dates and training isolation."""
import json
from pathlib import Path
import pandas as pd
import pytest
from market_forecasting.data import validate_prices, load_yahoo_snapshot, load_training_panel


def payload():
    """Return a two-session Yahoo-shaped fixture with distinct Close fields."""
    return {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1672689600, 1672776000],
        "indicators": {"quote": [{"open": [10., 11.], "high": [12., 13.],
          "low": [9., 10.], "close": [11., 12.], "volume": [100, 200]}],
          "adjclose": [{"adjclose": [10.5, 11.5]}]}}]}}


def write_sample(tmp_path):
    """Write the synthetic snapshot inside pytest's temporary directory; return its path."""
    p=tmp_path/'sample.json'; p.write_text(json.dumps(payload())); return p


def test_exchange_date_and_separate_adjusted_close(tmp_path):
    """Check exchange-local dates and preservation of distinct price conventions."""
    frame=load_yahoo_snapshot(write_sample(tmp_path))
    assert frame.index[0] == pd.Timestamp('2023-01-02')
    assert frame['close'].iloc[0] == 11
    assert frame['adj_close'].iloc[0] == 10.5


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'range', 'volume'])
def test_invalid_records_rejected(tmp_path, fault):
    """Ensure duplicates, nonfinite data and invalid financial ranges are rejected."""
    frame=load_yahoo_snapshot(write_sample(tmp_path))
    if fault=='duplicate': frame.index=[frame.index[0]]*2; frame.index=pd.DatetimeIndex(frame.index)
    if fault=='missing': frame.iloc[0, frame.columns.get_loc('close')]=float('nan')
    if fault=='range': frame.iloc[0, frame.columns.get_loc('high')]=1
    if fault=='volume': frame.iloc[0, frame.columns.get_loc('volume')]=-1
    with pytest.raises(ValueError): validate_prices(frame)


def test_training_panel_excludes_future_and_rejects_alignment(tmp_path):
    """Verify training slicing and rejection of misaligned instrument calendars."""
    raw=tmp_path/'data/raw'; raw.mkdir(parents=True)
    for symbol in ['NVDA','SPY']:
        (raw/f'{symbol}_yahoo_snapshot.json').write_text(json.dumps(payload()))
    config={'target_ticker':'NVDA','context_tickers':['SPY'],'start':'2023-01-01','train_end':'2023-01-02'}
    panel=load_training_panel(tmp_path,config)
    assert all(len(f)==1 and f.index.max()==pd.Timestamp('2023-01-02') for f in panel.values())
    modified=payload(); modified['chart']['result'][0]['timestamp']=[1672776000,1672862400]
    (raw/'SPY_yahoo_snapshot.json').write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='Trading dates differ|No observations'):
        load_training_panel(tmp_path,config)



def bounded_payload():
    """Return three synthetic sessions with two observations inside the boundary."""
    result = payload()["chart"]["result"][0]
    result["timestamp"] = [int(t.timestamp()) for t in pd.to_datetime([
        "2020-01-02 21:00Z", "2020-01-03 21:00Z", "2020-01-06 21:00Z"])]
    result["indicators"]["quote"][0] = {
        "open": [10., 11., 12.], "high": [12., 13., 14.],
        "low": [9., 10., 11.], "close": [11., 12., 13.], "volume": [100, 200, 300]}
    result["indicators"]["adjclose"][0]["adjclose"] = [10.5, 11.5, 12.5]
    return {"chart": {"error": None, "result": [result]}}


def bounded_file(tmp_path, value=None):
    path = tmp_path / "bounded.json"
    path.write_text(json.dumps(bounded_payload() if value is None else value))
    return path


def test_bounded_matches_full_synthetic_loader(tmp_path):
    path = bounded_file(tmp_path)
    expected = load_yahoo_snapshot(path).loc["2020-01-02":"2020-01-03"]
    actual = load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("key", ["open", "high", "low", "close", "volume", "adjclose"])
@pytest.mark.parametrize("tail", [None, -99999., "UNREAD_FUTURE_TOKEN"])
def test_future_quote_values_are_not_decoded(tmp_path, monkeypatch, key, tail):
    import market_forecasting.data as data_module
    expected = load_yahoo_snapshot(bounded_file(tmp_path), start="2020-01-02", end="2020-01-03")
    fixture = bounded_payload()
    indicators = fixture["chart"]["result"][0]["indicators"]
    if key == "adjclose":
        indicators["adjclose"][0][key][-1] = tail
    else:
        indicators["quote"][0][key][-1] = tail
    path = bounded_file(tmp_path, fixture)
    decode = data_module._number
    observed = []
    def guarded_decode(token, field):
        observed.append((field, token))
        if field != "timestamp":
            assert token not in {b'"UNREAD_FUTURE_TOKEN"', b"-99999.0", b"null"}
        return decode(token, field)
    monkeypatch.setattr(data_module, "_number", guarded_decode)
    actual = load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")
    pd.testing.assert_frame_equal(actual, expected)
    assert sum(field != "timestamp" for field, _ in observed) == 12


def test_unused_invalid_json_numeric_token_is_opaque(tmp_path):
    path = bounded_file(tmp_path)
    path.write_text(path.read_text().replace('"close": [11.0, 12.0, 13.0]',
                                             '"close": [11.0, 12.0, INVALID_TAIL]'))
    frame = load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")
    assert frame.close.tolist() == [11., 12.]


def test_earlier_excluded_prices_are_not_decoded(tmp_path, monkeypatch):
    import market_forecasting.data as data_module
    fixture = bounded_payload()
    fixture["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = "UNREAD_EARLIER_TOKEN"
    path = bounded_file(tmp_path, fixture)
    decode = data_module._number
    def guarded_decode(token, field):
        assert b"UNREAD_EARLIER_TOKEN" not in token
        return decode(token, field)
    monkeypatch.setattr(data_module, "_number", guarded_decode)
    assert load_yahoo_snapshot(path, start="2020-01-03", end="2020-01-03").close.tolist() == [12.]


@pytest.mark.parametrize("value", [None, -1., float("inf"), "invalid"])
def test_selected_invalid_close_rejected(tmp_path, value):
    fixture = bounded_payload()
    fixture["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = value
    with pytest.raises(ValueError):
        load_yahoo_snapshot(bounded_file(tmp_path, fixture), start="2020-01-02", end="2020-01-03")


@pytest.mark.parametrize("fault", ["short", "missing", "duplicate_empty", "duplicate_malformed", "timezone", "error"])
def test_malformed_bounded_schema_rejected(tmp_path, fault):
    fixture = bounded_payload()
    result = fixture["chart"]["result"][0]
    if fault == "short": result["indicators"]["quote"][0]["close"] = [11.]
    if fault == "missing": del result["indicators"]["quote"][0]["close"]
    if fault == "timezone": del result["meta"]["exchangeTimezoneName"]
    if fault == "error": fixture["chart"]["error"] = {"code": "Bad Request"}
    path = bounded_file(tmp_path, fixture)
    if fault.startswith("duplicate"):
        extra = '"close": [], ' if fault == "duplicate_empty" else '"close": [INVALID], '
        path.write_text(path.read_text().replace('"close": [', extra + '"close": [', 1))
    with pytest.raises(ValueError):
        load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")


@pytest.mark.parametrize("timestamps", [[1577998800,1577998800,1578344400],
                                        [1578085200,1577998800,1578344400]])
def test_bounded_dates_are_not_silently_reordered(tmp_path, timestamps):
    fixture = bounded_payload()
    fixture["chart"]["result"][0]["timestamp"] = timestamps
    with pytest.raises(ValueError, match="unique and increasing"):
        load_yahoo_snapshot(bounded_file(tmp_path, fixture), start="2020-01-01", end="2020-01-06")


@pytest.mark.parametrize("kwargs", [{"start":"2020-01-02"}, {"end":"2020-01-03"},
    {"start":"2020-01-03","end":"2020-01-02"}, {"start":"2020-02-01","end":"2020-02-02"},
    {"start":"2020-01-02 10:00","end":"2020-01-03"}])
def test_invalid_or_empty_intervals_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_yahoo_snapshot(bounded_file(tmp_path), **kwargs)


def test_complete_snapshot_without_later_timestamp(tmp_path):
    path = bounded_file(tmp_path)
    pd.testing.assert_frame_equal(load_yahoo_snapshot(path),
        load_yahoo_snapshot(path, start="2020-01-01", end="2020-01-31"))


def test_local_midnight_and_dst_conversion(tmp_path):
    fixture = bounded_payload()
    fixture["chart"]["result"][0]["timestamp"] = [int(t.timestamp()) for t in pd.to_datetime([
        "2020-03-07 00:30Z", "2020-03-10 00:30Z", "2020-03-11 00:30Z"])]
    path = bounded_file(tmp_path, fixture)
    frame = load_yahoo_snapshot(path, start="2020-03-06", end="2020-03-09")
    assert frame.index.tolist() == [pd.Timestamp("2020-03-06"), pd.Timestamp("2020-03-09")]
    pd.testing.assert_frame_equal(frame, load_yahoo_snapshot(path).iloc[:2])


def test_dates_only_reader_never_decodes_quotes(tmp_path, monkeypatch):
    import market_forecasting.data as data_module
    decode = data_module._number
    def dates_only(token, field):
        assert field == "timestamp"
        return decode(token, field)
    monkeypatch.setattr(data_module, "_number", dates_only)
    dates = data_module.load_yahoo_session_dates(bounded_file(tmp_path), start="2020-01-02", end="2020-01-03")
    assert dates.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_source_records_and_shared_quality_counts(tmp_path):
    from market_forecasting.data import load_yahoo_records, price_quality
    path = bounded_file(tmp_path)
    records = load_yahoo_records(path, start="2020-01-02", end="2020-01-03")
    frame = load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")
    assert records["indicators"]["quote"][0]["close"] == frame.close.tolist()
    assert price_quality(frame)["nonfinite_cells"] == 0
    frame.loc[frame.index[0], "close"] = float("nan")
    assert price_quality(frame)["nonfinite_cells"] == 1
    with pytest.raises(ValueError, match="non-finite"):
        validate_prices(frame)


def test_aligned_series_still_need_calendar_audit(tmp_path):
    from market_forecasting.calendar import audit_sessions
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    fixture = bounded_payload()
    result = fixture["chart"]["result"][0]
    # Remove the same real session from every series; cross-series equality survives.
    result["timestamp"].pop(1)
    for values in result["indicators"]["quote"][0].values(): values.pop(1)
    result["indicators"]["adjclose"][0]["adjclose"].pop(1)
    for symbol in ["NVDA", "SPY"]:
        (raw / f"{symbol}_yahoo_snapshot.json").write_text(json.dumps(fixture))
    config = {"target_ticker":"NVDA", "context_tickers":["SPY"], "start":"2020-01-02", "train_end":"2020-01-06"}
    panel = load_training_panel(tmp_path, config)
    audit = audit_sessions(panel["NVDA"].index, config["start"], config["train_end"], "NASDAQ")
    assert audit["missing"] == ["2020-01-03"]


def test_identity_verification_does_not_deserialize_prices(tmp_path, monkeypatch):
    import hashlib
    import market_forecasting.data as data_module
    path = bounded_file(tmp_path)
    record = {"retrieved_at_utc":"2026-10-11T00:00:00+00:00", "series":{"NVDA":{
        "splits":[{"date":"2020-01-03", "ratio":"4:1", "close":"UNREAD_AUDIT_PRICE"}],
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}}}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(record))
    def forbidden(*args, **kwargs):
        raise AssertionError("Identity and event metadata must not deserialize numerical history")
    monkeypatch.setattr(data_module.json, "loads", forbidden)
    assert data_module.snapshot_provenance(path, audit_path, "NVDA")["checksum_matches"]
    assert data_module.recorded_splits(audit_path, "NVDA", start="2020-01-02", end="2020-01-03") == [
        {"date":"2020-01-03", "ratio":"4:1"}]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data_module.snapshot_provenance(path, audit_path, "NVDA")


def test_bounded_reader_never_deserializes_a_full_document(tmp_path, monkeypatch):
    import market_forecasting.data as data_module
    path = bounded_file(tmp_path)
    decode = data_module.json.loads
    def scalar_only(value, *args, **kwargs):
        assert isinstance(value, bytes)
        assert value == b"null" or data_module._NUMBER.fullmatch(value)
        return decode(value, *args, **kwargs)
    monkeypatch.setattr(data_module.json, "loads", scalar_only)
    assert len(load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03")) == 2


def test_bounded_reader_preserves_vendor_column_order(tmp_path):
    fixture = bounded_payload()
    indicators = fixture["chart"]["result"][0]["indicators"]
    indicators["quote"][0] = dict(reversed(list(indicators["quote"][0].items())))
    path = bounded_file(tmp_path, fixture)
    pd.testing.assert_frame_equal(load_yahoo_snapshot(path).iloc[:2],
        load_yahoo_snapshot(path, start="2020-01-02", end="2020-01-03"))
