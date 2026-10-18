"""Causal technical indicators and common-eligibility residual LSTM datasets."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .preprocessing import Standardisation
from .calendar import audit_sessions

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


def calculate_features(series, config, *, history_end=None):
    """Build causal indicators through an explicit historical boundary."""
    end = config["validation_end"] if history_end is None else history_end
    if history_end is not None:
        end = _history_boundary(config, end, config["test_end"])
    closes = {}
    for ticker in ("NVDA", "SPY", "XLK"):
        s = series[ticker]
        if not isinstance(s.index, pd.DatetimeIndex) or s.index.has_duplicates or not s.index.is_monotonic_increasing:
            raise ValueError("Unique chronological session dates required")
        # Bound the prefix before conversion; the default remains development-only.
        s = s.loc[config["start"]:end].astype(float)
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


def _validate_feature_configuration(config):
    """Require the established price convention, lookback and chronological design."""
    if config["price_convention"] != "split_adjusted_close" or config["forecast_horizon_trading_days"] != 1:
        raise ValueError("Expected split-adjusted next-session Close")
    if config["initial_sequence_length"] != 30:
        raise ValueError("Milestone 6 lookback is fixed at 30")
    if not pd.Timestamp(config["start"]) <= pd.Timestamp(config["train_end"]) < pd.Timestamp(config["validation_end"]) < pd.Timestamp(config["test_end"]):
        raise ValueError("Invalid chronological boundaries")


def _prepare_feature_dataset(features, config, variant, fit_end, evaluation_end,
                             require_validation):
    """Construct shared eligible windows with transforms fitted on the chosen labels."""
    _validate_feature_configuration(config)
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("Unique chronological feature dates required")
    frame = features.loc[config["start"]:evaluation_end, VARIANTS["C"]]
    complete = np.isfinite(frame.to_numpy()).all(axis=1)
    # Enforce the documented warm-up, then reject gaps instead of dropping sessions.
    if len(frame) <= 79 or complete[:49].any() or not complete[49:].all():
        raise ValueError("Expected 49 warm-up rows followed by complete features")
    frame = frame.iloc[49:]
    length = config["initial_sequence_length"]
    dates = frame.index
    metadata = pd.DataFrame({"window_start": dates[:-length], "origin_date": dates[length-1:-1],
                             "target_date": dates[length:]})
    fit = (metadata.target_date <= pd.Timestamp(fit_end)).to_numpy()
    if not fit.any() or (require_validation and fit.all()):
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


def prepare_feature_dataset(features, config, variant):
    """Use the same complete-feature interval and target dates for every variant."""
    return _prepare_feature_dataset(features, config, variant, config["train_end"],
                                    config["validation_end"], require_validation=True)


def _pretest_boundary(config, end):
    """Reject dates outside the explicitly authorised pre-test period."""
    end = pd.Timestamp(end)
    if (pd.isna(end) or end.tz is not None or end != end.normalize()
            or not pd.Timestamp(config["start"]) <= end
            <= min(pd.Timestamp(config["validation_end"]), pd.Timestamp("2024-12-31"))):
        raise ValueError("Final preparation is restricted to pre-test sessions through 2024-12-31")
    return end


def _check_session_index(index, start, end, calendar_name, label):
    """Reject incomplete or inconsistent session history before numerical processing."""
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or index.has_duplicates or not index.is_monotonic_increasing
            or not index.equals(index.normalize())):
        raise ValueError(f"Unique chronological daily session dates required for {label}")
    audit = audit_sessions(index, start, end, calendar_name)
    if audit["missing"] or audit["unexpected"] or not len(index):
        raise ValueError(f"Missing or unexpected session dates for {label}")


def prepare_final_feature_dataset(features, config, variant, fit_end):
    """Fit transforms and construct every eligible pre-test target without validation."""
    end = _pretest_boundary(config, fit_end)
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Final features require daily session dates")
    # Restrict the rows before any array conversion can inspect a reserved value.
    prefix = features.loc[config["start"]:end]
    _check_session_index(prefix.index, config["start"], end, "NASDAQ", "final features")
    data = _prepare_feature_dataset(prefix.astype(float), config, variant, end, end,
                                    require_validation=False)
    validate_feature_preprocessing(data.provenance, variant)
    return data


def validate_feature_preprocessing(preprocessing, variant):
    """Validate frozen final-fit feature order, sample boundaries and scaler metadata."""
    if variant not in VARIANTS or not isinstance(preprocessing, dict):
        raise ValueError("Invalid final preprocessing variant or metadata")
    names = VARIANTS[variant]
    if preprocessing.get("feature_names") != names:
        raise ValueError("Frozen preprocessing feature order does not match the model variant")
    integer_keys = ("warmup_rows", "input_unique_rows", "fit_examples", "validation_examples")
    if any(not isinstance(preprocessing.get(key), int) or isinstance(preprocessing.get(key), bool)
           for key in integer_keys):
        raise ValueError("Frozen preprocessing counts must be integers")
    if (preprocessing["warmup_rows"] != 49 or preprocessing["fit_examples"] < 1
            or preprocessing["validation_examples"] != 0
            or preprocessing["input_unique_rows"] != preprocessing["fit_examples"] + 29):
        raise ValueError("Frozen preprocessing warm-up or unique-input counts are inconsistent")
    dates = []
    for key in ("first_feature_date", "first_target_date", "last_fitting_input_date",
                "last_fitting_target_date"):
        try:
            value = pd.Timestamp(preprocessing[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid frozen preprocessing date: {key}") from error
        if (pd.isna(value) or value.tz is not None or value != value.normalize()
                or value > pd.Timestamp("2024-12-31")):
            raise ValueError(f"Invalid pre-test preprocessing date: {key}")
        dates.append(value)
    first_feature, first_target, last_input, last_target = dates
    if not (first_feature < first_target <= last_target and first_feature <= last_input < last_target):
        raise ValueError("Frozen preprocessing input/target boundaries are inconsistent")
    scalers = preprocessing.get("input_scalers")
    if not isinstance(scalers, dict) or set(scalers) != set(names):
        raise ValueError("Frozen preprocessing input scaler names do not match the feature order")
    for name, scaler in [*scalers.items(), ("correction", preprocessing.get("correction_scaler"))]:
        if not isinstance(scaler, dict) or set(scaler) != {"mean", "scale"}:
            raise ValueError(f"Invalid frozen scaler metadata for {name}")
        try:
            valid = (not isinstance(scaler["mean"], bool) and not isinstance(scaler["scale"], bool)
                     and np.isfinite(scaler["mean"]) and np.isfinite(scaler["scale"])
                     and scaler["scale"] > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"Frozen scaler for {name} must have a finite mean and positive finite scale")


def _history_boundary(config, origin, permitted_end):
    """Validate an explicit historical limit without changing dataset partitions."""
    origin, limit = pd.Timestamp(origin), pd.Timestamp(permitted_end)
    if any(pd.isna(date) or date.tz is not None or date != date.normalize()
           for date in (origin, limit)):
        raise ValueError("Historical boundaries must be timezone-naive daily dates")
    if not pd.Timestamp(config["start"]) <= origin <= limit <= pd.Timestamp(config["test_end"]):
        raise ValueError("Forecast origin exceeds the permitted historical boundary")
    return origin


def prepare_origin_input(series, config, variant, origin, preprocessing, *, permitted_end=None):
    """Transform a historical origin window without requiring its unknown target."""
    _validate_feature_configuration(config)
    # The caller must authorise evaluation before supplying a later explicit limit.
    end = (_pretest_boundary(config, origin) if permitted_end is None
           else _history_boundary(config, origin, permitted_end))
    validate_feature_preprocessing(preprocessing, variant)
    bounded = {}
    for ticker, calendar_name in (("NVDA", "NASDAQ"), ("SPY", "NYSE"), ("XLK", "NYSE")):
        if not isinstance(series[ticker].index, pd.DatetimeIndex):
            raise ValueError(f"Daily session dates required for {ticker}")
        # Slicing precedes float conversion, so later prices cannot enter indicators.
        prefix = series[ticker].loc[config["start"]:end]
        _check_session_index(prefix.index, config["start"], end, calendar_name, ticker)
        if prefix.index[-1] != end:
            raise ValueError("Forecast origin must be an observed trading session")
        bounded[ticker] = prefix
    # Each call preserves the original indicator seeds by retaining the full prefix.
    _, features = calculate_features(bounded, config, history_end=end)
    complete = np.isfinite(features[VARIANTS["C"]].to_numpy()).all(axis=1)
    if len(features) < 79 or complete[:49].any() or not complete[49:].all():
        raise ValueError("Insufficient continuous feature history for the 30-session origin window")
    if str(features.index[49].date()) != preprocessing["first_feature_date"]:
        raise ValueError("Historical feature start does not match frozen preprocessing")
    window = features.iloc[-30:]
    names = VARIANTS[variant]
    transformed = np.column_stack([
        Standardisation(**preprocessing["input_scalers"][name]).transform(window[name])
        for name in names])
    X = transformed[np.newaxis, :, :].astype("float32")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite transformed forecast-origin inputs")
    metadata = {"window_start": str(window.index[0].date()), "origin_date": str(end.date()),
                "feature_names": list(names), "lookback": 30}
    return X, float(window.close.iloc[-1]), metadata
