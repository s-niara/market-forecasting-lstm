"""Aligned forecast evaluation and paired uncertainty calculations for frozen models."""
import numpy as np
import pandas as pd

from market_forecasting.calendar import audit_sessions
from market_forecasting.dataset import benchmark_relative_metrics

SEEDS = (42, 123, 2026)
VARIANTS = ("A", "B", "C")
FORECAST_COLUMNS = ("naive", "ARIMA") + tuple(
    f"{variant}_seed_{seed}" for variant in VARIANTS for seed in SEEDS
)
REQUIRED_COLUMNS = ("origin_date", "target_date", "origin_close", "actual_close") + FORECAST_COLUMNS
COMPARISONS = (("A", "naive"), ("ARIMA", "naive"), ("B", "A"), ("C", "B"))


def validate_evaluation_table(table):
    """Validate one complete, contiguous target-date sequence shared by every model."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Evaluation requires a nonempty pandas DataFrame")
    if table.columns.has_duplicates or set(table.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Evaluation columns must match the approved metadata and eleven forecasts")
    frame = table.loc[:, REQUIRED_COLUMNS].copy()
    for name in ("origin_date", "target_date"):
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(frame[name], errors="raise"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name} dates") from exc
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError(f"{name} must contain timezone-naive daily session labels")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"{name} must be unique and chronological")
        frame[name] = dates
    origins = pd.DatetimeIndex(frame["origin_date"])
    targets = pd.DatetimeIndex(frame["target_date"])
    if not (origins < targets).all() or not origins[1:].equals(targets[:-1]):
        raise ValueError("Forecast origins must precede targets and continue from the previous target")
    # Calendar validation also checks the first origin-to-target step; a missed
    # session must not silently turn a one-session forecast into a longer horizon.
    coverage = audit_sessions(origins[:1].append(targets), origins[0], targets[-1], "NASDAQ")
    if coverage["missing"] or coverage["unexpected"] or coverage["duplicate_dates"]:
        raise ValueError("Evaluation dates must cover every consecutive NASDAQ session")
    for name in ("origin_close", "actual_close") + FORECAST_COLUMNS:
        try:
            values = pd.to_numeric(frame[name], errors="raise").to_numpy()
            if np.iscomplexobj(values) or values.dtype.kind == "b":
                raise ValueError("Boolean and complex prices are not valid")
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numeric observations") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        if name in ("origin_close", "actual_close") and (values <= 0).any():
            raise ValueError(f"{name} must contain positive closing prices")
        frame[name] = values
    if not np.array_equal(frame["naive"].to_numpy(), frame["origin_close"].to_numpy()):
        raise ValueError("Naive forecasts must equal the latest observed Close")
    if not np.array_equal(frame["origin_close"].to_numpy()[1:], frame["actual_close"].to_numpy()[:-1]):
        raise ValueError("Each subsequent origin Close must equal the preceding observed target")
    return frame.reset_index(drop=True)


def forecast_errors(table):
    """Return aligned signed USD errors, with positive values denoting overprediction."""
    frame = validate_evaluation_table(table)
    errors = pd.DataFrame({"target_date": frame["target_date"]})
    with np.errstate(over="raise", invalid="raise"):
        try:
            for name in FORECAST_COLUMNS:
                errors[name] = frame[name].to_numpy() - frame["actual_close"].to_numpy()
        except FloatingPointError as exc:
            raise ValueError("Forecast errors exceed finite numerical limits") from exc
    return errors


def _relative_skills(mae, rmse, naive):
    """Express aggregate errors relative to the same persistence benchmark."""
    absolute, percentage = {}, {}
    for label, value in (("mae", mae), ("rmse", rmse)):
        baseline = naive[f"{label}_usd"]
        absolute[label] = baseline - value
        percentage[label] = 100 * (1 - value / baseline) if baseline else None
    return {"absolute_skill_usd": absolute, "percentage_skill": percentage}


def evaluate_predictions(table):
    """Calculate seed metrics first, then family aggregates and prespecified contrasts."""
    frame = validate_evaluation_table(table)
    actual, naive = frame["actual_close"].to_numpy(), frame["naive"].to_numpy()
    per_model = {}
    with np.errstate(over="raise", invalid="raise"):
        try:
            for name in FORECAST_COLUMNS:
                result = benchmark_relative_metrics(actual, frame[name].to_numpy(), naive)
                per_model[name] = {**result["model"],
                                   "absolute_skill_usd": result["absolute_skill_usd"],
                                   "percentage_skill": result["percentage_skill"]}
        except FloatingPointError as exc:
            raise ValueError("Evaluation metrics exceed finite numerical limits") from exc
    families = {}
    for variant in VARIANTS:
        seed_metrics = [per_model[f"{variant}_seed_{seed}"] for seed in SEEDS]
        maes = np.array([item["mae_usd"] for item in seed_metrics])
        rmses = np.array([item["rmse_usd"] for item in seed_metrics])
        # Each seed has its own error calculation. Averaging predictions first
        # would introduce an unapproved ensemble and change the RMSE estimand.
        families[variant] = {
            "n": len(frame), "seeds": list(SEEDS),
            "mean_mae_usd": float(maes.mean()), "mean_rmse_usd": float(rmses.mean()),
            "sample_std_mae_usd": float(maes.std(ddof=1)),
            "sample_std_rmse_usd": float(rmses.std(ddof=1)),
            **_relative_skills(float(maes.mean()), float(rmses.mean()), per_model["naive"]),
        }
    paired = {}
    for left, right in COMPARISONS:
        differences = {}
        # Pair matching seeds for LSTM contrasts; repeat the fixed benchmark
        # against each seed only when the first model is an LSTM family.
        for seed in SEEDS if left in VARIANTS else (None,):
            left_key = f"{left}_seed_{seed}" if left in VARIANTS else left
            right_key = f"{right}_seed_{seed}" if right in VARIANTS else right
            differences[str(seed) if seed is not None else "benchmark"] = {
                f"{metric}_difference_usd": per_model[left_key][f"{metric}_usd"]
                - per_model[right_key][f"{metric}_usd"] for metric in ("mae", "rmse")
            }
        entry = {"first": left, "second": right,
                 "negative_difference_favours": left,
                 "per_seed": differences if left in VARIANTS else {}}
        for metric in ("mae", "rmse"):
            values = np.array([item[f"{metric}_difference_usd"] for item in differences.values()])
            entry[f"mean_{metric}_difference_usd"] = float(values.mean())
            entry[f"sample_std_{metric}_difference_usd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        paired[f"{left}-{right}"] = entry
    return {
        "n": len(frame), "target_start": str(frame["target_date"].iloc[0].date()),
        "target_end": str(frame["target_date"].iloc[-1].date()),
        "error_convention": "prediction_minus_actual",
        "family_aggregation": "arithmetic_mean_of_individual_seed_metrics_no_prediction_ensemble",
        "seed_variability": "sample_standard_deviation_ddof_1",
        "per_model": per_model, "families": families, "paired_comparisons": paired,
    }


def paired_circular_bootstrap(table, block_lengths=(10, 5, 20), repetitions=5000, seed=314159):
    """Estimate approximate paired MAE-difference intervals using shared circular blocks."""
    frame = validate_evaluation_table(table)
    if (not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 2
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("Bootstrap repetitions must be at least two and the seed a nonnegative integer")
    try:
        lengths = tuple(block_lengths)
    except TypeError as exc:
        raise ValueError("Block lengths must be a nonempty sequence of positive integers") from exc
    if (not lengths or any(not isinstance(length, int) or isinstance(length, bool) or length < 1
                           for length in lengths) or len(set(lengths)) != len(lengths)):
        raise ValueError("Block lengths must be unique positive integers")
    if max(lengths) > len(frame):
        raise ValueError("At least the largest block length of observations is required; do not shrink the protocol")
    errors = forecast_errors(frame).loc[:, FORECAST_COLUMNS].abs()
    family_errors = {name: errors[name].to_numpy() for name in ("naive", "ARIMA")}
    for variant in VARIANTS:
        # Mean absolute seed errors at each date yield the mean of seed MAEs
        # after date resampling; no predicted prices are averaged.
        family_errors[variant] = errors[[f"{variant}_seed_{seed}" for seed in SEEDS]].mean(axis=1).to_numpy()
    labels = [f"{left}-{right}" for left, right in COMPARISONS]
    differences = np.column_stack([family_errors[left] - family_errors[right] for left, right in COMPARISONS])
    rng = np.random.default_rng(seed)
    intervals = {}
    for length in lengths:
        count = int(np.ceil(len(frame) / length))
        draws = np.empty((repetitions, len(COMPARISONS)), dtype=float)
        for repetition in range(repetitions):
            starts = rng.integers(0, len(frame), size=count)
            # Modulo wrapping retains within-block neighbours; trim only the
            # final block so every draw contains exactly n jointly sampled dates.
            indices = ((starts[:, None] + np.arange(length)) % len(frame)).ravel()[:len(frame)]
            draws[repetition] = differences[indices].mean(axis=0)
        intervals[str(length)] = {
            label: {"estimate_usd": float(differences[:, index].mean()),
                    "lower_usd": float(np.quantile(draws[:, index], 0.025)),
                    "upper_usd": float(np.quantile(draws[:, index], 0.975))}
            for index, label in enumerate(labels)
        }
    return {
        "method": "paired_circular_block_bootstrap", "statistic": "difference_of_mean_seed_maes_usd",
        "n": len(frame), "block_lengths": list(lengths), "repetitions": repetitions,
        "seed": seed, "random_generator": "numpy.default_rng.PCG64",
        "interval_level": 0.95, "interval_method": "percentile_linear_quantile",
        "joint_date_resampling_across_models_and_seeds": True,
        "intervals": intervals,
        "limitations": [
            "Approximate conditional uncertainty for these frozen models and this observed period.",
            "Circular blocks assume within-period dependence is adequately represented at the chosen block lengths.",
            "The last and first observed dates are joined during resampling; regime changes can weaken this approximation.",
            "Seeds remain fixed; date resampling does not estimate uncertainty from a wider population of training seeds.",
            "Intervals are supplementary, unadjusted for multiple comparisons, and cannot select models or guarantee future performance.",
        ],
    }
