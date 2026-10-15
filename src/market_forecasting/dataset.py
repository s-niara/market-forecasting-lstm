"""Unscaled daily close windows. Test-period construction is deliberately disabled."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ForecastDataset:
    """Unscaled examples and temporal metadata shared by development benchmarks."""
    # X: (examples, lookback sessions, 1 Close feature); y: (examples,).
    # metadata keeps each window start, forecast origin, target date and partition.
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def build_development_dataset(close: pd.Series, config: dict) -> ForecastDataset:
    """Construct causal windows and assign partitions by forecast target date."""
    if config.get("price_convention") != "split_adjusted_close":
        raise ValueError("Target convention must be explicitly agreed")
    if config["forecast_horizon_trading_days"] != 1:
        raise ValueError("Only next-session forecasts are implemented")
    length = config["initial_sequence_length"]
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("Sequence length must be a positive integer")
    # Expect a Close Series with daily session labels. Ordering is checked here;
    # complete exchange-calendar coverage must have been audited separately.
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Unique chronological DatetimeIndex required")
    start, train_end, val_end = map(pd.Timestamp, [config["start"],config["train_end"],config["validation_end"]])
    if not start <= train_end < val_end < pd.Timestamp(config["test_end"]):
        raise ValueError("Invalid split boundaries")
    # Slice before numerical checks, window construction or statistics: no test prices enter this path.
    development = close.loc[start:val_end]
    values = development.to_numpy(dtype=float)
    if len(values) <= length or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Insufficient or invalid development prices")
    dates = development.index
    # Each slice stops before its target i. Add a feature axis for the
    # (examples, sessions, features) shape expected by an LSTM.
    X=np.stack([values[i-length:i] for i in range(length,len(values))])[...,None]
    y=values[length:].copy()
    targets=dates[length:]
    # Window starts, final input dates and target dates have equal lengths;
    # partition by target date so boundary context is retained without leakage.
    metadata=pd.DataFrame({
        "window_start":dates[:-length], "origin_date":dates[length-1:-1],
        "target_date":targets,
        "split":np.where(targets <= train_end,"train","validation")})
    return ForecastDataset(X,y,metadata)


def naive_predictions(dataset: ForecastDataset) -> np.ndarray:
    """Return the last observed Close as each next-session forecast."""
    # Select the final session and sole feature, retaining the original USD scale.
    return dataset.X[:,-1,0].copy()


def error_metrics(actual, predicted) -> dict:
    """Calculate MAE and RMSE for aligned one-dimensional observations."""
    # Callers must align target dates and invert target scaling before supplying USD prices.
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size==0:
        raise ValueError("Expected equal nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    # Absolute errors retain dollar units; squaring before averaging emphasises large misses.
    errors=predicted-actual
    return {"n":int(len(actual)), "mae_usd":float(np.mean(np.abs(errors))),
            "rmse_usd":float(np.sqrt(np.mean(errors**2)))}
