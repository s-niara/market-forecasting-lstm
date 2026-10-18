"""Causal technical indicators and common-eligibility residual LSTM datasets."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .preprocessing import Standardisation

TECHNICAL = ["sma20_distance", "sma50_distance", "ema12_distance", "ema26_distance",
             "rsi14_fraction", "macd_relative", "histogram_relative"]
VARIANTS = {"A": ["close"], "B": ["close", *TECHNICAL],
            "C": ["close", *TECHNICAL, "spy_log_return", "xlk_log_return"]}


def seeded_ema(values, period):
    """Seed an EMA with the first period's mean, retaining its missing prefix."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    start = int(valid[0])
    if not np.isfinite(values[start:]).all():
        raise ValueError("EMA input has an internal missing value")
    if len(values) - start < period:
        return result
    first = start + period - 1
    result[first] = values[start:first + 1].mean()
    alpha = 2 / (period + 1)
    for i in range(first + 1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def wilder_rsi(close, period=14):
    """Calculate RSI using an arithmetic seed and Wilder's recursive smoothing."""
    values = np.asarray(close, dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) <= period:
        return result
    changes = np.diff(values)
    gains, losses = np.maximum(changes, 0), np.maximum(-changes, 0)
    gain, loss = gains[:period].mean(), losses[:period].mean()
    for i in range(period, len(values)):
        if i > period:
            gain = ((period - 1) * gain + gains[i - 1]) / period
            loss = ((period - 1) * loss + losses[i - 1]) / period
        # Flat prices are neutral; one-sided changes have defined extreme values.
        result[i] = 50 if gain == loss == 0 else (100 if loss == 0 else 100 * gain / (gain + loss))
    return result


def calculate_features(series, config):
    """Build aligned development-only indicators without filling unavailable history."""
    closes = {}
    for ticker in ("NVDA", "SPY", "XLK"):
        s = series[ticker]
        if not isinstance(s.index, pd.DatetimeIndex) or s.index.has_duplicates or not s.index.is_monotonic_increasing:
            raise ValueError("Unique chronological session dates required")
        # Exclude 2025 before any indicator or numerical validity calculation.
        s = s.loc[config["start"]:config["validation_end"]].astype(float)
        if not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError("Missing, nonfinite or nonpositive development Close")
        closes[ticker] = s
    close = closes["NVDA"]
    if any(not s.index.equals(close.index) for s in closes.values()):
        raise ValueError("Instrument sessions must match exactly")
    raw = pd.DataFrame({"close": close})
    for n in (20, 50):
        raw[f"sma{n}"] = close.rolling(n, min_periods=n).mean()
    for n in (12, 26):
        raw[f"ema{n}"] = seeded_ema(close, n)
    raw["rsi14"] = wilder_rsi(close)
    raw["macd"] = raw.ema12 - raw.ema26
    raw["signal"] = seeded_ema(raw.macd, 9)
    raw["histogram"] = raw.macd - raw.signal
    features = pd.DataFrame({"close": close})
    for name in ("sma20", "sma50", "ema12", "ema26"):
        features[f"{name}_distance"] = close / raw[name] - 1
    features["rsi14_fraction"] = raw.rsi14 / 100
    features["macd_relative"] = raw.macd / close
    features["histogram_relative"] = raw.histogram / close
    for ticker in ("SPY", "XLK"):
        # shift(1) refers only to the previous observed session, not a future quote.
        features[f"{ticker.lower()}_log_return"] = np.log(closes[ticker] / closes[ticker].shift(1))
    return raw, features


@dataclass
class FeatureDataset:
    """Keep scaled sequences with unscaled prices, dates and fitted transformations."""
    X: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    fit_mask: np.ndarray
    correction_scaler: Standardisation
    provenance: dict


def prepare_feature_dataset(features, config, variant):
    """Use the same complete-feature interval and target dates for every variant."""
    if config["price_convention"] != "split_adjusted_close" or config["forecast_horizon_trading_days"] != 1:
        raise ValueError("Expected split-adjusted next-session Close")
    if config["initial_sequence_length"] != 30:
        raise ValueError("Milestone 6 lookback is fixed at 30")
    if not pd.Timestamp(config["start"]) <= pd.Timestamp(config["train_end"]) < pd.Timestamp(config["validation_end"]) < pd.Timestamp(config["test_end"]):
        raise ValueError("Invalid chronological boundaries")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("Unique chronological feature dates required")
    frame = features.loc[config["start"]:config["validation_end"], VARIANTS["C"]]
    complete = np.isfinite(frame.to_numpy()).all(axis=1)
    # Enforce the documented warm-up, then reject gaps instead of dropping sessions.
    if len(frame) <= 79 or complete[:49].any() or not complete[49:].all():
        raise ValueError("Expected 49 warm-up rows followed by complete features")
    frame = frame.iloc[49:]
    length = config["initial_sequence_length"]
    dates = frame.index
    metadata = pd.DataFrame({"window_start": dates[:-length], "origin_date": dates[length-1:-1],
                             "target_date": dates[length:]})
    fit = (metadata.target_date <= pd.Timestamp(config["train_end"])).to_numpy()
    if not fit.any() or fit.all():
        raise ValueError("Both training and validation examples are required")
    metadata["split"] = np.where(fit, "train", "validation")
    names = VARIANTS[variant]
    # The last fitting target is not an input unless it appears in an earlier fitting window.
    unique = frame.loc[:metadata.loc[fit, "origin_date"].max(), names]
    scalers = {name: Standardisation.fit(unique[name]) for name in names}
    scaled = np.column_stack([scalers[name].transform(frame[name]) for name in names])
    # Shape is (examples, 30 sessions, 1/8/10 features); the target row is excluded.
    X = np.stack([scaled[i-length:i] for i in range(length, len(frame))]).astype("float32")
    origins = frame.close.to_numpy()[length-1:-1].copy()
    targets = frame.close.to_numpy()[length:].copy()
    corrections = targets - origins
    target_scaler = Standardisation.fit(corrections[fit])
    provenance = {"feature_names": names, "warmup_rows": 49, "input_unique_rows": len(unique),
                  "first_feature_date": str(dates[0].date()),
                  "first_target_date": str(metadata.target_date.iloc[0].date()),
                  "last_fitting_input_date": str(unique.index[-1].date()),
                  "last_fitting_target_date": str(metadata.loc[fit, "target_date"].iloc[-1].date()),
                  "fit_examples": int(fit.sum()), "validation_examples": int((~fit).sum()),
                  "input_scalers": {name: asdict(scaler) for name, scaler in scalers.items()},
                  "correction_scaler": asdict(target_scaler)}
    return FeatureDataset(X, target_scaler.transform(corrections).astype("float32"), origins,
                          targets, metadata, fit, target_scaler, provenance)
