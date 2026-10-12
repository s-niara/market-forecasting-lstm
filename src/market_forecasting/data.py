"""Read frozen Yahoo responses and enforce basic financial-data invariants.

Close is the vendor's split-adjusted series. Adjusted close is retained separately;
we do not silently apply dividend adjustments or re-adjust volume here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_prices(frame: pd.DataFrame) -> None:
    required = ["open", "high", "low", "close", "volume", "adj_close"]
    if frame.empty:
        raise ValueError("No observations")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite prices/volume")
    if (frame[["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if ((frame["low"] > frame[["open", "close"]].min(axis=1)) |
        (frame["high"] < frame[["open", "close"]].max(axis=1)) |
        (frame["low"] > frame["high"])).any():
        raise ValueError("Inconsistent OHLC range")


def load_yahoo_snapshot(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo response contains an error or no result")
    result = chart["result"][0]
    timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    dates = dates.tz_convert(timezone).normalize().tz_localize(None)
    frame = pd.DataFrame(result["indicators"]["quote"][0], index=dates)
    frame["adj_close"] = result["indicators"]["adjclose"][0]["adjclose"]
    frame.index.name = "date"
    validate_prices(frame)
    return frame


def load_training_panel(root: str | Path, config: dict) -> dict[str, pd.DataFrame]:
    """Return training-period data only, rejecting misaligned series."""
    root = Path(root)
    symbols = [config["target_ticker"], *config["context_tickers"]]
    panel = {s: load_yahoo_snapshot(root / "data/raw" / f"{s}_yahoo_snapshot.json")
             .loc[config["start"]:config["train_end"]].copy() for s in symbols}
    reference = panel[symbols[0]].index
    if reference.empty:
        raise ValueError("Training interval is empty")
    if any(not frame.index.equals(reference) for frame in panel.values()):
        raise ValueError("Trading dates differ; resolve alignment before modelling")
    return panel
