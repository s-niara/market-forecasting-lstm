"""Read frozen Yahoo histories within explicit dates and validate daily records."""
import hashlib
import json
import mmap
import re
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_FIELDS = ["open", "high", "low", "close", "adj_close"]
REQUIRED_FIELDS = ["open", "high", "low", "close", "volume", "adj_close"]
# Frozen chart arrays contain scalar numbers or null, not nested JSON values.
_ARRAY_TOKEN = re.compile(rb'\s*([^,\]\s]+)\s*([,\]])')
_NUMBER = re.compile(rb'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z')


def price_quality(frame: pd.DataFrame) -> dict:
    """Return the diagnostic counts used by the financial-data validator."""
    if frame.empty:
        raise ValueError("No observations")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Expected a DatetimeIndex without missing dates")
    missing = set(REQUIRED_FIELDS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    values = frame[REQUIRED_FIELDS].to_numpy(dtype=float)
    # A valid session range contains both the opening and closing price.
    range_bad = ((frame.low > frame[["open", "close"]].min(axis=1)) |
                 (frame.high < frame[["open", "close"]].max(axis=1)) |
                 (frame.low > frame.high))
    return {
        "null_cells": int(frame[REQUIRED_FIELDS].isna().sum().sum()),
        "nonfinite_cells": int((~np.isfinite(values)).sum()),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "ordered": bool(frame.index.is_monotonic_increasing),
        "nonpositive_prices": int((frame[PRICE_FIELDS] <= 0).sum().sum()),
        "negative_volume": int((frame.volume < 0).sum()),
        "zero_volume": int((frame.volume == 0).sum()),
        "invalid_OHLC_rows": int(range_bad.sum()),
    }


def validate_prices(frame: pd.DataFrame) -> None:
    """Reject malformed daily price records without modifying the frame."""
    checks = price_quality(frame)
    if checks["duplicate_dates"] or not checks["ordered"]:
        raise ValueError("Dates must be unique and increasing")
    if checks["nonfinite_cells"]:
        raise ValueError("Missing or non-finite prices/volume")
    if checks["nonpositive_prices"]:
        raise ValueError("Prices must be positive")
    if checks["negative_volume"]:
        raise ValueError("Volume cannot be negative")
    if checks["invalid_OHLC_rows"]:
        raise ValueError("Inconsistent OHLC range")


def _bounds(start, end):
    """Validate inclusive, timezone-naive session boundaries."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(first) or pd.isna(last) or first.tz is not None or last.tz is not None
            or first != first.normalize() or last != last.normalize() or first > last):
        raise ValueError("Use ordered, timezone-naive session dates for start and end")
    return first, last


def _unique_match(pattern, buffer, name):
    """Reject missing or ambiguous fields in the frozen snapshot schema."""
    matches = list(re.finditer(pattern, buffer))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} in the frozen schema")
    return matches[0]


def _timezone(buffer):
    """Read only the exchange timezone and response-status fields."""
    _unique_match(rb'"chart"\s*:\s*\{', buffer, "chart object")
    _unique_match(rb'"result"\s*:\s*\[\s*\{', buffer, "result object")
    error = _unique_match(rb'"error"\s*:\s*([^,}\s]+)', buffer, "error field")
    if error[1] != b"null":
        raise ValueError("Yahoo response contains an error")
    match = _unique_match(rb'"exchangeTimezoneName"\s*:\s*"([^"\\]+)"', buffer, "exchange timezone")
    return match[1].decode("utf-8")


def _array_tokens(buffer, key):
    """Yield raw scalar tokens without interpreting excluded price values."""
    pattern = rb'"' + key.encode() + rb'"\s*:\s*\['
    if key == "adjclose":
        # Yahoo nests the numeric adjclose array inside an outer object array.
        pattern += rb'(?!\s*\{)'
    match = _unique_match(pattern, buffer, f"{key} array")
    position = match.end()
    if re.match(rb'\s*\]', buffer[position:position + 64]):
        return
    while True:
        match = _ARRAY_TOKEN.match(buffer, position)
        if match is None:
            raise ValueError(f"Malformed requested {key} array prefix")
        yield match[1]
        if match[2] == b"]":
            return
        position = match.end()


def _number(token, key):
    """Decode a selected numeric value, retaining null for financial validation."""
    if token != b"null" and _NUMBER.fullmatch(token) is None:
        raise ValueError(f"Invalid numeric token in selected {key} records")
    return json.loads(token)


def _session_dates(timestamps, timezone):
    """Convert UTC timestamps to exchange-local daily session labels."""
    dates = pd.to_datetime(timestamps, unit="s", utc=True)
    return dates.tz_convert(timezone).normalize().tz_localize(None).rename("date")


def _selected_timestamps(buffer, start, end, timezone):
    """Locate the requested interval in the frozen chronological timestamp prefix."""
    timestamps, positions = [], []
    previous = None
    for position, token in enumerate(_array_tokens(buffer, "timestamp")):
        stamp = _number(token, "timestamp")
        if not isinstance(stamp, int):
            raise ValueError("Expected integer Unix timestamps")
        day = pd.Timestamp(stamp, unit="s", tz="UTC").tz_convert(timezone).normalize().tz_localize(None)
        if previous is not None and day <= previous:
            raise ValueError("Dates must be unique and increasing")
        # Only the first later date is inspected to locate the stopping boundary.
        # This cannot certify ordering/completeness in an intentionally unread tail.
        if day > end:
            break
        previous = day
        if day >= start:
            timestamps.append(stamp)
            positions.append(position)
    if not timestamps:
        raise ValueError("No observations in requested interval")
    return timestamps, positions[0], positions[-1] + 1


def load_yahoo_records(path: str | Path, *, start, end) -> dict:
    """Read unchanged Yahoo fields for an inclusive interval in a frozen snapshot."""
    first, last = _bounds(start, end)
    # Locate fields as bytes; never deserialize the full response or price metadata.
    # This reader deliberately supports the frozen scalar-array schema only. It
    # relies on chronological acquisition history; unused tails are not validated.
    with Path(path).open("rb") as handle:
        if Path(path).stat().st_size == 0:
            raise ValueError("Empty snapshot")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            timezone = _timezone(buffer)
            timestamps, lower, upper = _selected_timestamps(buffer, first, last, timezone)
            fields = {}
            quote_keys = ("open", "high", "low", "close", "volume")
            # Keep the vendor's column order as well as its numerical values.
            quote_keys = sorted(quote_keys, key=lambda key: _unique_match(
                rb'"' + key.encode() + rb'"\s*:\s*\[', buffer, f"{key} array").start())
            for key in (*quote_keys, "adjclose"):
                # Skip earlier tokens lexically; decode exactly the selected values.
                tokens = list(islice(_array_tokens(buffer, key), lower, upper))
                if len(tokens) != len(timestamps):
                    raise ValueError(f"Short {key} array in requested interval")
                fields[key] = [_number(token, key) for token in tokens]
    return {"meta": {"exchangeTimezoneName": timezone}, "timestamp": timestamps,
            "indicators": {"quote": [{key: fields[key] for key in quote_keys}],
                           "adjclose": [{"adjclose": fields["adjclose"]}]}}


def load_yahoo_session_dates(path: str | Path, *, start, end) -> pd.DatetimeIndex:
    """Read bounded session labels without decoding any quote values."""
    first, last = _bounds(start, end)
    with Path(path).open("rb") as handle:
        if Path(path).stat().st_size == 0:
            raise ValueError("Empty snapshot")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            timezone = _timezone(buffer)
            timestamps, _, _ = _selected_timestamps(buffer, first, last, timezone)
    return _session_dates(timestamps, timezone)


def load_yahoo_snapshot(path: str | Path, *, start=None, end=None) -> pd.DataFrame:
    """Load validated prices, restricting value decoding when both bounds are given."""
    if (start is None) != (end is None):
        raise ValueError("Specify both start and end, or neither")
    if start is not None:
        result = load_yahoo_records(path, start=start, end=end)
    else:
        # Retain the historical administrative API; research callers supply bounds.
        chart = json.loads(Path(path).read_text())["chart"]
        if chart.get("error") or not chart.get("result"):
            raise ValueError("Yahoo response contains an error or no result")
        result = chart["result"][0]
    timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    frame = pd.DataFrame(result["indicators"]["quote"][0], index=_session_dates(result["timestamp"], timezone))
    # Preserve the two vendor fields without applying another adjustment.
    frame["adj_close"] = result["indicators"]["adjclose"][0]["adjclose"]
    validate_prices(frame)
    return frame


def load_training_panel(root: str | Path, config: dict) -> dict[str, pd.DataFrame]:
    """Return validated, aligned training frames without decoding later prices."""
    root = Path(root)
    symbols = [config["target_ticker"], *config["context_tickers"]]
    panel = {symbol: load_yahoo_snapshot(root / "data/raw" / f"{symbol}_yahoo_snapshot.json",
                                       start=config["start"], end=config["train_end"])
             for symbol in symbols}
    reference = panel[symbols[0]].index
    # Agreement does not detect sessions missing from every instrument; callers
    # must also demonstrate coverage with the independent calendar audit.
    if any(not frame.index.equals(reference) for frame in panel.values()):
        raise ValueError("Trading dates differ; resolve alignment before modelling")
    return panel


def _audit_series(audit, symbol):
    """Locate one acquisition record without interpreting its numerical fields."""
    series = audit.split(b'"date_alignment"', 1)[0]
    pattern = rb'"' + re.escape(symbol.encode()) + rb'"\s*:\s*\{'
    start = _unique_match(pattern, series, f"{symbol} acquisition record").end() - 1
    depth = 0
    # Quoted strings are skipped as complete tokens; braces within them are data.
    for token in re.finditer(rb'"(?:[^"\\]|\\.)*"|[{}]', series[start:]):
        if token[0] == b"{":
            depth += 1
        elif token[0] == b"}":
            depth -= 1
            if depth == 0:
                return series[start:start + token.end()]
    raise ValueError(f"Unterminated acquisition record for {symbol}")


def snapshot_provenance(path: str | Path, audit_path: str | Path, symbol: str) -> dict:
    """Verify opaque file identity and return acquisition identifiers only."""
    # The historical audit also contains full-period statistics. Extract only the
    # identifiers needed here instead of deserializing that numerical report.
    audit = Path(audit_path).read_bytes()
    retrieved = _unique_match(rb'"retrieved_at_utc"\s*:\s*"([^"]+)"', audit, "retrieval timestamp")[1].decode()
    record = _audit_series(audit, symbol)
    expected = _unique_match(rb'"sha256"\s*:\s*"([a-f0-9]{64})"', record, "acquisition checksum")[1].decode()
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch: {symbol}")
    return {"instrument": symbol, "snapshot": Path(path).name,
            "retrieved_utc": retrieved, "checksum_matches": True, "sha256": digest}


def recorded_splits(audit_path: str | Path, symbol: str, *, start, end) -> list[dict]:
    """Extract bounded split dates and ratios without decoding recorded prices."""
    first, last = _bounds(start, end)
    audit = Path(audit_path).read_bytes()
    record = _audit_series(audit, symbol)
    splits = _unique_match(rb'"splits"\s*:\s*\[([\s\S]*?)\]', record, "split-event list")
    events = []
    for match in re.finditer(rb'"date"\s*:\s*"([^"]+)"\s*,\s*"ratio"\s*:\s*"([^"]+)"', splits[1]):
        date = match[1].decode()
        if first <= pd.Timestamp(date) <= last:
            events.append({"date": date, "ratio": match[2].decode()})
    return events
