"""Training-only standardisation and residual-price reconstruction."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .dataset import build_development_dataset


@dataclass(frozen=True)
class Standardisation:
    """Store the fitting mean and population standard deviation."""
    mean: float
    scale: float

    @classmethod
    def fit(cls, values):
        """Estimate statistics from the supplied fitting observations only."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected a nonempty finite fitting vector")
        scale = float(values.std(ddof=0))
        # A constant fitting vector carries no variance; unit scale avoids division by zero.
        return cls(float(values.mean()), scale if scale > 0 else 1.0)

    def transform(self, values):
        """Apply stored statistics without learning from new observations."""
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values):
        """Restore the units used when fitting the transform."""
        return np.asarray(values, dtype=float) * self.scale + self.mean


def reconstruct_prices(origins, standardised_corrections, scaler):
    """Add inverse-transformed dollar corrections to observed origin prices."""
    origins = np.asarray(origins, dtype=float)
    corrections = np.asarray(standardised_corrections, dtype=float)
    if origins.ndim != 1 or corrections.shape != origins.shape:
        raise ValueError("Origin prices and corrections must be aligned vectors")
    if not np.isfinite(origins).all() or not np.isfinite(corrections).all():
        raise ValueError("Nonfinite reconstruction input")
    # The origin is known at forecast time; the target price must never enter this sum.
    return origins + scaler.inverse_transform(corrections)


@dataclass
class ResidualData:
    """Keep scaled arrays alongside original prices and target-date metadata."""
    X: np.ndarray
    y: np.ndarray
    fit_mask: np.ndarray
    dataset: object
    input_scaler: Standardisation
    correction_scaler: Standardisation
    provenance: dict


def prepare_residual_data(close, config, fit_end, evaluation_end):
    """Build stage-specific windows and fit transforms only on fitting examples."""
    if not (pd.Timestamp(config["start"]) < pd.Timestamp(fit_end)
            < pd.Timestamp(evaluation_end) <= pd.Timestamp(config["validation_end"])):
        raise ValueError("Invalid fitting/evaluation boundaries")
    stage = dict(config, train_end=fit_end, validation_end=evaluation_end)
    # Truncate before numerical processing, even if the caller supplied later prices.
    close = close.loc[config["start"]:evaluation_end]
    dataset = build_development_dataset(close, stage)
    fit_mask = dataset.metadata.split.eq("train").to_numpy()
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("Both fitting and evaluation examples are required")
    fitting = dataset.metadata.loc[fit_mask]
    # Fit on each input date once, not the repeated values in overlapping windows.
    unique_inputs = close.loc[fitting.window_start.min():fitting.origin_date.max()]
    input_scaler = Standardisation.fit(unique_inputs.to_numpy())
    origins = dataset.X[:, -1, 0]
    corrections = dataset.y - origins
    correction_scaler = Standardisation.fit(corrections[fit_mask])
    provenance = {
        "fit_end": fit_end, "evaluation_end": evaluation_end,
        "input_first_date": str(unique_inputs.index.min().date()),
        "input_last_date": str(unique_inputs.index.max().date()),
        "input_unique_rows": len(unique_inputs), "fit_examples": int(fit_mask.sum()),
        "evaluation_examples": int((~fit_mask).sum()),
        "target_first_date": str(fitting.target_date.min().date()),
        "target_last_date": str(fitting.target_date.max().date()),
        "input_scaler": asdict(input_scaler), "correction_scaler": asdict(correction_scaler),
    }
    # LSTM inputs are (examples, sessions, features); labels remain a vector.
    return ResidualData(input_scaler.transform(dataset.X).astype("float32"),
                        correction_scaler.transform(corrections).astype("float32"),
                        fit_mask, dataset, input_scaler, correction_scaler, provenance)
