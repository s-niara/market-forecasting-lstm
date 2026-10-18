"""Visualise aligned forecasts without loading data, fitting models or exporting files."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .evaluation import FORECAST_COLUMNS, SEEDS, VARIANTS, evaluate_predictions, forecast_errors, validate_evaluation_table


COLORS = {"naive": "#64748b", "ARIMA": "#8b5cf6", "A": "#2563eb", "B": "#d97706", "C": "#059669"}
REPRESENTATIVE_SEED = 42
REPRESENTATIVE_COLUMNS = ("naive", "ARIMA", *(f"{variant}_seed_42" for variant in VARIANTS))


def _family(column: str) -> str:
    """Return the family identifier used for a forecast column."""
    return column.split("_seed_")[0]


def _label(column: str) -> str:
    """Label individual LSTM runs explicitly rather than implying an ensemble."""
    return column.replace("naive", "Naïve").replace("_seed_", " · seed ")


def _style(ax, *, dates: bool = False) -> None:
    """Apply consistent local formatting without changing global plotting defaults."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    if dates:
        ax.margins(x=0)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _validated(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the canonical table through the authoritative error implementation."""
    frame = validate_evaluation_table(table)
    errors = forecast_errors(frame)
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame, errors


def _period(frame: pd.DataFrame) -> str:
    """Describe the actual target range supplied to the plotting function."""
    dates = frame.target_date
    return f"{dates.iloc[0]:%d %b %Y}–{dates.iloc[-1]:%d %b %Y}; {len(frame):,} aligned targets"


def plot_actual_predictions(table: pd.DataFrame) -> Figure:
    """Plot actual prices and the prespecified seed-42 forecasts on shared dates."""
    frame, _ = _validated(table)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(frame.target_date, frame.actual_close, color="#0f172a", linewidth=2, label="Actual Close")
    for column in REPRESENTATIVE_COLUMNS:
        ax.plot(frame.target_date, frame[column], color=COLORS[_family(column)], linewidth=1.1,
                alpha=0.85, label=_label(column))
    ax.set(title=f"Actual versus forecast NVIDIA Close\n{_period(frame)}",
           xlabel="Target session", ylabel="Split-adjusted Close (USD)")
    ax.legend(ncol=3, fontsize=9)
    _style(ax, dates=True)
    return fig


def plot_signed_errors(table: pd.DataFrame) -> Figure:
    """Show forecast-minus-actual errors separately for each representative model."""
    frame, errors = _validated(table)
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, sharey=True, layout="constrained")
    for ax, column in zip(axes, REPRESENTATIVE_COLUMNS):
        ax.plot(frame.target_date, errors[column], color=COLORS[_family(column)], linewidth=1)
        ax.axhline(0, color="#334155", linewidth=0.7)
        ax.set(ylabel="Error (USD)", title=_label(column))
        _style(ax, dates=True)
    axes[-1].set_xlabel("Target session")
    fig.suptitle(f"Signed errors: forecast − actual (positive = overprediction)\n{_period(frame)}", fontsize=12)
    return fig


def plot_absolute_error_distributions(table: pd.DataFrame) -> Figure:
    """Compare every seed's absolute-error distribution without pooling predictions."""
    frame, errors = _validated(table)
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    values = [errors[column].abs().to_numpy() for column in FORECAST_COLUMNS]
    labels = [_label(column).replace(" · seed ", "\nseed ") for column in FORECAST_COLUMNS]
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True,
                     medianprops={"color": "#0f172a"}, flierprops={"markersize": 3, "alpha": 0.45})
    for patch, column in zip(box["boxes"], FORECAST_COLUMNS):
        patch.set(facecolor=COLORS[_family(column)], alpha=0.35)
    ax.set(title=f"Absolute forecasting errors: benchmarks and all LSTM seeds\n{_period(frame)}",
           ylabel="Absolute error (USD)", xlabel="Model / random seed")
    _style(ax)
    return fig


def plot_naive_relative_performance(table: pd.DataFrame) -> Figure:
    """Plot absolute and percentage skill using means of seed-specific metrics."""
    frame, _ = _validated(table)
    results = evaluate_predictions(frame)
    labels = ("ARIMA", "A", "B", "C")
    rows = [results["per_model"]["ARIMA"], *(results["families"][variant] for variant in VARIANTS)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row_index, metric in enumerate(("mae", "rmse")):
        for column_index, key in enumerate(("absolute_skill_usd", "percentage_skill")):
            ax = axes[row_index, column_index]
            values = [row[key][metric] for row in rows]
            for position, (label, value) in enumerate(zip(labels, values)):
                if value is None:
                    # A zero naïve error makes percentage skill undefined, not zero.
                    ax.text(position, 0.5, "Undefined\n(naïve error = 0)", ha="center", va="center",
                            transform=ax.get_xaxis_transform(), fontsize=8)
                else:
                    ax.bar(position, value, color=COLORS[label], alpha=0.8)
            ax.set_xticks(range(len(labels)), ["ARIMA", "A mean", "B mean", "C mean"])
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set(title=f"{metric.upper()}: {'absolute improvement' if column_index == 0 else 'percentage skill'}",
                   ylabel="Naïve − model error (USD)" if column_index == 0 else "Skill relative to naïve (%)")
            _style(ax)
    fig.suptitle(f"Naïve-relative performance: positive values indicate improvement\n{_period(frame)}", fontsize=12)
    return fig


def plot_seed_variability(table: pd.DataFrame) -> Figure:
    """Show all seed metrics with their family means and sample standard deviations."""
    frame, _ = _validated(table)
    results = evaluate_predictions(frame)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    markers = ("o", "s", "^")
    offsets = (-0.1, 0, 0.1)
    for ax, metric in zip(axes, ("mae", "rmse")):
        for position, variant in enumerate(VARIANTS):
            for seed, marker, offset in zip(SEEDS, markers, offsets):
                value = results["per_model"][f"{variant}_seed_{seed}"][f"{metric}_usd"]
                ax.scatter(position + offset, value, marker=marker, color=COLORS[variant], s=44, zorder=3)
            summary = results["families"][variant]
            ax.errorbar(position + 0.23, summary[f"mean_{metric}_usd"],
                        yerr=summary[f"sample_std_{metric}_usd"], fmt="_", color="#0f172a",
                        capsize=4, markersize=10, zorder=4)
        ax.axhline(results["per_model"]["naive"][f"{metric}_usd"], color=COLORS["naive"], linestyle="--")
        ax.set(xticks=range(3), xticklabels=["A", "B", "C"], xlim=(-0.4, 2.5),
               title=metric.upper(), ylabel="Error (USD)", xlabel="LSTM feature variant")
        _style(ax)
    handles = [Line2D([], [], color="#334155", marker=marker, linestyle="", label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, markers)]
    handles += [Line2D([], [], color="#0f172a", marker="_", linestyle="", label="Mean ± seed SD"),
                Line2D([], [], color=COLORS["naive"], linestyle="--", label="Naïve")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle(f"LSTM seed variability: SD is not a confidence interval\n{_period(frame)}", fontsize=12)
    return fig


def plot_paired_differences(table: pd.DataFrame) -> Figure:
    """Show the prespecified feature contrasts with paired seeds and their mean."""
    frame, _ = _validated(table)
    results = evaluate_predictions(frame)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    markers = ("o", "s", "^")
    for ax, metric in zip(axes, ("mae", "rmse")):
        for position, comparison in enumerate(("B-A", "C-B")):
            row = results["paired_comparisons"][comparison]
            color = COLORS[comparison[0]]
            for offset, seed, marker in zip((-0.12, 0, 0.12), SEEDS, markers):
                ax.scatter(position + offset, row["per_seed"][str(seed)][f"{metric}_difference_usd"],
                           color=color, marker=marker, s=48)
            ax.scatter(position + 0.27, row[f"mean_{metric}_difference_usd"], color="#0f172a", marker="_", s=160)
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set(xticks=(0, 1), xticklabels=("B − A", "C − B"), xlim=(-0.4, 1.6),
               title=metric.upper(), ylabel="Paired error difference (USD)", xlabel="Feature comparison")
        _style(ax)
    handles = [Line2D([], [], color="#334155", marker=marker, linestyle="", label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, markers)]
    handles += [Line2D([], [], color="#0f172a", marker="_", linestyle="", label="Mean difference")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Paired feature comparisons: negative values favour the first model\n{_period(frame)}", fontsize=12)
    return fig


def plot_cumulative_error_differences(table: pd.DataFrame) -> Figure:
    """Accumulate paired absolute-error differences without averaging forecasts."""
    frame, errors = _validated(table)
    absolute = errors.loc[:, FORECAST_COLUMNS].abs()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    comparisons = (("A", "naive"), ("ARIMA", "naive"), ("B", "A"), ("C", "B"))
    for ax, (left, right) in zip(axes.flat, comparisons):
        differences = []
        seeds = SEEDS if left in VARIANTS else (None,)
        for seed in seeds:
            left_column = f"{left}_seed_{seed}" if left in VARIANTS else left
            right_column = f"{right}_seed_{seed}" if right in VARIANTS else right
            differences.append((absolute[left_column] - absolute[right_column]).to_numpy())
        cumulative = np.cumsum(np.stack(differences), axis=1)
        if len(seeds) > 1:
            for values in cumulative:
                ax.plot(frame.target_date, values, color=COLORS[left], alpha=0.25, linewidth=0.9)
        # Averaging paired losses is distinct from constructing an ensemble prediction.
        ax.plot(frame.target_date, cumulative.mean(axis=0), color=COLORS[left], linewidth=1.8,
                label="Mean across paired seeds" if len(seeds) > 1 else "ARIMA − naïve")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set(title=f"{left} − {'Naïve' if right == 'naive' else right}",
               ylabel="Cumulative absolute-error difference (USD)", xlabel="Target session")
        ax.legend(fontsize=8)
        _style(ax, dates=True)
    fig.suptitle(f"Cumulative paired losses: below zero favours the first model\n{_period(frame)}", fontsize=12)
    return fig


def select_forecast_examples(table: pd.DataFrame) -> list[pd.Timestamp]:
    """Select the first target and three largest naïve-error targets with chronological ties."""
    frame, errors = _validated(table)
    ranked = pd.DataFrame({"target_date": frame.target_date.to_numpy(), "error": errors.naive.abs().to_numpy()})
    ranked = ranked.sort_values(["error", "target_date"], ascending=[False, True], kind="stable")
    candidates = [frame.target_date.iloc[0], *ranked.target_date.iloc[:3].tolist()]
    # The first target may already be among the three largest misses; show it only once.
    return list(dict.fromkeys(candidates))


def plot_forecast_examples(table: pd.DataFrame) -> Figure:
    """Compare every representative model on the same prespecified illustrative dates."""
    frame, errors = _validated(table)
    dates = select_forecast_examples(frame)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, target in zip(axes.flat, dates):
        row = frame.loc[frame.target_date == target].iloc[0]
        for position, column in enumerate(REPRESENTATIVE_COLUMNS):
            value = row[column]
            ax.scatter(position, value, color=COLORS[_family(column)], s=65, zorder=4)
            ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=COLORS[_family(column)])
        ax.axhline(row.actual_close, color="#0f172a", linewidth=1.4, linestyle="--")
        ax.axhline(row.origin_close, color="#64748b", linewidth=1, linestyle=":")
        reason = "First target" if target == frame.target_date.iloc[0] else "Large naïve miss"
        naive_error = float(errors.loc[frame.target_date == target, "naive"].abs().iloc[0])
        ax.set(xticks=range(5), xticklabels=("Naïve", "ARIMA", "A · 42", "B · 42", "C · 42"),
               xlim=(-0.5, 4.5), ylabel="Split-adjusted Close (USD)",
               title=f"{target:%d %b %Y} · {reason}\nOrigin: {row.origin_date:%d %b %Y}; naïve |error| = ${naive_error:.2f}")
        ax.margins(y=0.28)
        _style(ax)
    for ax in axes.flat[len(dates):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color="#0f172a", linestyle="--", label="Actual target Close"),
               Line2D([], [], color="#64748b", linestyle=":", label="Observed origin Close")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle("Shared forecast examples: all LSTM predictions use seed 42\nFirst target + three largest naïve errors; duplicate dates shown once", fontsize=12)
    return fig
