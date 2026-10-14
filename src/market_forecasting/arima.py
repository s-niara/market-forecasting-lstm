"""Training-selected ARIMA and causal, fixed-parameter validation forecasts."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .dataset import build_development_dataset, error_metrics


@dataclass
class ArimaExperiment:
    """Validation predictions and reproducibility evidence for one ARIMA experiment."""
    # One row per validation target, with actual Close and both forecasts.
    predictions: pd.DataFrame
    summary: dict


def select_training_model(training, orders, maxiter=500):
    """Select a converged ARIMA candidate using training AIC only."""
    # The caller supplies training prices only; validation cannot influence AIC selection.
    values = np.asarray(training, dtype=float)
    if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all():
        raise ValueError("At least 30 finite training prices are required")
    if not orders:
        raise ValueError("At least one candidate order is required")
    records, best, best_key = [], None, None
    for raw_order in orders:
        order = tuple(raw_order)
        if (len(order) != 3 or any(type(v) is not int or v < 0 for v in order)
                or order[1] != 1):
            raise ValueError("Candidate orders must be nonnegative integer (p, 1, q)")
        record = {"order": list(order), "eligible": False}
        # Preserve optimiser warnings as evidence rather than silently accepting
        # a problematic fit. Nonconverged candidates cannot win selection.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # Fit coefficients on the differenced price history, without drift.
                # maxiter caps optimisation; a lower AIC need not improve forecasts.
                fit = ARIMA(values, order=order, trend="n",
                            enforce_stationarity=True, enforce_invertibility=True).fit(
                                method_kwargs={"maxiter": maxiter})
                converged = bool(fit.mle_retvals.get("converged", False))
                finite = bool(np.isfinite(fit.aic) and np.isfinite(fit.params).all())
                record.update(converged=converged,
                              aic=float(fit.aic) if np.isfinite(fit.aic) else None,
                              eligible=converged and finite)
                if record["eligible"]:
                    # Deterministic tie-breaking favours a smaller order.
                    key = (float(fit.aic), order[0] + order[2], order)
                    if best_key is None or key < best_key:
                        best, best_key = fit, key
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["warnings"] = sorted({str(w.message) for w in caught})
        records.append(record)
    if best is None:
        raise RuntimeError(f"No converged finite ARIMA candidate: {records}")
    return best, records


def rolling_one_step(fitted, observed):
    """Forecast before each observed target, then update state with fixed parameters."""
    # The fitted history must end just before these chronological target observations.
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("Finite one-dimensional validation observations required")
    state, predictions = fitted, []
    for actual in observed:
        # Forecast before revealing this target; otherwise its actual value would leak
        # into its own prediction. Extract the single horizon as a scalar.
        forecast = float(np.asarray(state.forecast(steps=1)).reshape(-1)[0])
        if not np.isfinite(forecast):
            raise RuntimeError("Non-finite ARIMA forecast; run aborted")
        predictions.append(forecast)
        # Filter the newly observed Close for the next forecast, without refitting coefficients.
        state = state.extend([float(actual)])
    return np.asarray(predictions)


def run_arima_experiment(close, config, settings):
    """Compare ARIMA and naive forecasts on identical development target dates."""
    dataset = build_development_dataset(close, config)
    if settings.get("selection") != "training_aic" or settings.get("refit") is not False:
        raise ValueError("Only training-AIC selection with fixed parameters is supported")
    if settings.get("trend") != "n":
        raise ValueError("This benchmark requires no deterministic trend")
    # Separate permissible fitting observations from validation targets. The
    # existing builder has already checked conventions and chronological ordering.
    development = close.loc[config["start"]:config["validation_end"]].astype(float)
    training = development.loc[:config["train_end"]]
    validation = development.loc[development.index > pd.Timestamp(config["train_end"])]
    mask = dataset.metadata["split"].eq("validation")
    targets = pd.DatetimeIndex(dataset.metadata.loc[mask, "target_date"])
    if validation.empty or not targets.equals(validation.index):
        raise ValueError("Validation targets must match the existing window dataset exactly")
    # ARIMA uses the full training history, while the naive comparator uses windows.
    # Neither order selection nor coefficient fitting receives validation observations.
    fitted, candidates = select_training_model(
        training.to_numpy(), settings["orders"], settings["maxiter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecasts = rolling_one_step(fitted, validation.to_numpy())
    # Keep both models on the same origins/targets before computing USD errors.
    table = dataset.metadata.loc[mask, ["origin_date", "target_date"]].reset_index(drop=True)
    table["actual_close"] = validation.to_numpy()
    table["arima_prediction"] = forecasts
    table["naive_prediction"] = dataset.X[mask.to_numpy(), -1, 0]
    arima_metrics = error_metrics(table.actual_close, table.arima_prediction)
    naive_metrics = error_metrics(table.actual_close, table.naive_prediction)
    summary = {
        "model": "ARIMA", "selected_order": list(fitted.model.order),
        "selection": "lowest converged training AIC; no validation-based order selection",
        "trend": "n", "refit": False, "state_update": "forecast then extend observed target",
        "training_rows": len(training), "training_start": str(training.index.min().date()),
        "training_end": str(training.index.max().date()),
        "validation_start": str(validation.index.min().date()),
        "validation_end": str(validation.index.max().date()),
        "target": "Yahoo split-adjusted Close, retrospective USD basis",
        "parameters": dict(zip(fitted.param_names, map(float, fitted.params))),
        "candidates": candidates,
        "forecast_warnings": sorted({str(w.message) for w in caught}),
        "validation_metrics": {"arima": arima_metrics, "naive": naive_metrics},
        "mae_improvement_pct": 100 * (1 - arima_metrics["mae_usd"] / naive_metrics["mae_usd"])
            if naive_metrics["mae_usd"] else None,
        "test_evaluated": False,
    }
    return ArimaExperiment(table, summary)
