"""Training-selected ARIMA and causal, fixed-parameter validation forecasts."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .dataset import build_development_dataset, error_metrics
from .calendar import audit_sessions


@dataclass
class ArimaExperiment:
    """Validation predictions and reproducibility evidence for one ARIMA experiment."""
    # One row per validation target, with actual Close and both forecasts.
    predictions: pd.DataFrame
    summary: dict



def _fit_order(values, order, maxiter):
    """Fit one no-drift specification using the established numerical settings."""
    return ARIMA(values, order=order, trend="n",
                 enforce_stationarity=True, enforce_invertibility=True).fit(
                     method_kwargs={"maxiter": maxiter})


def fit_fixed_order(training, order=(2, 1, 2), maxiter=500):
    """Fit a frozen ARIMA order and reject invalid or nonconverged outcomes."""
    values = np.asarray(training, dtype=float)
    order = tuple(order)
    if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all():
        raise ValueError("At least 30 finite training prices are required")
    if (len(order) != 3 or any(type(v) is not int or v < 0 for v in order)
            or order[1] != 1):
        raise ValueError("Fixed order must be nonnegative integer (p, 1, q)")
    if type(maxiter) is not int or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    # This entry point fits exactly one prespecified order; no AIC search occurs.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = _fit_order(values, order, maxiter)
    converged = bool(fitted.mle_retvals.get("converged", False))
    finite = bool(np.isfinite(fitted.aic) and np.isfinite(fitted.params).all()
                  and np.isfinite(fitted.filter_results.filtered_state).all())
    evidence = {"order": list(order), "trend": "n", "maxiter": maxiter,
                "enforce_stationarity": True, "enforce_invertibility": True,
                "training_rows": len(values), "converged": converged, "finite": finite,
                "aic": float(fitted.aic) if np.isfinite(fitted.aic) else None,
                "parameters": dict(zip(fitted.param_names, map(float, fitted.params))),
                "warnings": sorted({str(w.message) for w in caught})}
    if not converged or not finite:
        raise RuntimeError(f"Fixed ARIMA fit failed verification: {evidence}")
    return fitted, evidence


def _session_date(value):
    """Require an exchange-local, timezone-naive session label."""
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tz is not None or date != date.normalize():
        raise ValueError("Expected a timezone-naive session date at midnight")
    return date


class FrozenArima:
    """Enforce forecast-then-observe sequencing with fixed fitted coefficients."""
    def __init__(self, fitted, last_observed_date, *, permitted_end="2024-12-31"):
        self.state = fitted
        self.last_observed_date = _session_date(last_observed_date)
        # An evaluation caller supplies its authorised boundary; the default stays pre-test.
        self.permitted_end = _session_date(permitted_end)
        if self.last_observed_date > self.permitted_end:
            raise ValueError("ARIMA state extends beyond the permitted historical boundary")
        self.pending_target = None
        self.parameters = np.asarray(fitted.params, dtype=float).copy()
        if not np.isfinite(self.parameters).all():
            raise ValueError("Nonfinite fitted ARIMA parameters")

    def _check_parameters(self, state):
        if not np.array_equal(np.asarray(state.params), self.parameters):
            raise RuntimeError("Frozen ARIMA coefficients changed")

    def forecast(self, origin_date, target_date):
        """Issue one forecast without accepting the unknown target observation."""
        origin, target = _session_date(origin_date), _session_date(target_date)
        if target > self.permitted_end:
            raise ValueError("Forecast exceeds its permitted boundary; later dates require separate authorisation")
        if self.pending_target is not None:
            raise ValueError("Observe the pending target before issuing another forecast")
        if origin != self.last_observed_date or target <= origin:
            raise ValueError("Forecast origin must equal the last observed session")
        # A chronological gap is still invalid: each forecast must be one session ahead.
        calendar = audit_sessions(pd.DatetimeIndex([origin, target]), origin, target, "NASDAQ")
        if calendar["missing"] or calendar["unexpected"] or calendar["expected_sessions"] != 2:
            raise ValueError("Forecast target must be the next NASDAQ session")
        self._check_parameters(self.state)
        prediction = np.asarray(self.state.forecast(steps=1), dtype=float).reshape(-1)
        self._check_parameters(self.state)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise RuntimeError("Nonfinite or invalid one-step ARIMA forecast")
        self.pending_target = target
        return float(prediction[0])

    def observe(self, target_date, actual):
        """Filter the just-forecast observation without re-estimating coefficients."""
        target = _session_date(target_date)
        if self.pending_target is None or target != self.pending_target:
            raise ValueError("Observation must match an outstanding forecast target")
        actual = np.asarray(actual, dtype=float)
        if actual.ndim != 0 or not np.isfinite(actual) or actual <= 0:
            raise ValueError("Observed Close must be a finite positive scalar")
        self._check_parameters(self.state)
        updated = self.state.extend([float(actual)])
        self._check_parameters(self.state)
        self._check_parameters(updated)
        self.state, self.last_observed_date = updated, target
        self.pending_target = None


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
                fit = _fit_order(values, order, maxiter)
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


def rolling_one_step(fitted, observed, *, trace=None):
    """Forecast before each observed target, then update state with fixed parameters."""
    # The fitted history must end just before these chronological target observations.
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("Finite one-dimensional validation observations required")
    state, predictions = fitted, []
    for step, actual in enumerate(observed):
        # Forecast before revealing this target; otherwise its actual value would leak
        # into its own prediction. Extract the single horizon as a scalar.
        forecast = float(np.asarray(state.forecast(steps=1)).reshape(-1)[0])
        if not np.isfinite(forecast):
            raise RuntimeError("Non-finite ARIMA forecast; run aborted")
        predictions.append(forecast)
        if trace is not None:
            # Record the forecast before revealing its target; copy parameter values for inspection.
            trace.append({"step": step, "event": "forecast", "prediction": forecast,
                          "parameters": np.asarray(state.params).tolist()})
        # Filter the newly observed Close for the next forecast, without refitting coefficients.
        state = state.extend([float(actual)])
        if trace is not None:
            trace.append({"step": step, "event": "update", "observed_close": float(actual),
                          "parameters": np.asarray(state.params).tolist()})
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
