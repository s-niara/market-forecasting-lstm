"""Summarise saved Milestone 6 evidence and plot training-only feature relationships."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from market_forecasting.dataset import error_metrics

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"A": "#2563eb", "B": "#d97706", "C": "#059669"}


def aggregate(rows):
    """Summarise run metrics using means and sample standard deviations."""
    return {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                     "std": float(np.std([r[metric] for r in rows], ddof=1))}
            for metric in ("mae_usd", "rmse_usd")}


def main():
    """Read all nine runs without fitting models or calculating test-period features."""
    base = ROOT / "artifacts/features_v1"
    runs = json.loads((base / "runs.json").read_text())
    if len(runs) != 9 or any(r["status"] != "complete" for r in runs):
        raise ValueError("Nine complete runs required; retain failures without replacement")
    if {(r["variant"], r["seed"]) for r in runs} != {(v, s) for v in "ABC" for s in (42, 123, 2026)}:
        raise ValueError("Unexpected run identities")
    naive = pd.read_json(ROOT / "artifacts/naive_v1/validation_predictions.json")
    arima = pd.read_json(ROOT / "artifacts/arima_v1/validation_predictions.json")
    for table in (naive, arima):
        for column in ("origin_date", "target_date"):
            table[column] = pd.to_datetime(table[column])
    baselines = {"naive": json.loads((ROOT / "artifacts/naive_v1/summary.json").read_text())["validation_metrics"],
                 "arima": json.loads((ROOT / "artifacts/arima_v1/summary.json").read_text())["validation_metrics"]["arima"]}
    tables = {}
    for run in runs:
        name = f"{run['variant']}_seed_{run['seed']}"
        table = pd.read_json(base / name / "predictions.json")
        for column in ("origin_date", "target_date"):
            table[column] = pd.to_datetime(table[column])
        for other in (naive, arima):
            for column in ("target_date", "origin_date"):
                pd.testing.assert_series_equal(pd.to_datetime(table[column]).astype("datetime64[ns]"),
                                               pd.to_datetime(other[column]).astype("datetime64[ns]"), check_names=False)
            np.testing.assert_allclose(table.actual_close, other.actual_close, rtol=0, atol=1e-8)
        measured = error_metrics(table.actual_close, table.prediction)
        for key in ("mae_usd", "rmse_usd"):
            np.testing.assert_allclose(measured[key], run["evaluation_metrics"][key], rtol=0, atol=1e-8)
        assert run["epochs_completed"] == 19 and run["best_epoch"] is None and not run["stopped_early"]
        assert not run["test_evaluated"] and table.target_date.max().year == 2024
        tables[name] = table
    grouped = {v: aggregate([r["evaluation_metrics"] for r in runs if r["variant"] == v]) for v in "ABC"}
    annual = {str(y): {v: aggregate([r["by_year"][str(y)] for r in runs if r["variant"] == v])
                       for v in "ABC"} for y in (2023, 2024)}
    annual_baselines = {str(y): {"naive": error_metrics(part.actual_close, part.prediction),
                                "arima": error_metrics(arima.loc[arima.target_date.dt.year == y, "actual_close"],
                                                       arima.loc[arima.target_date.dt.year == y, "arima_prediction"])}
                        for y, part in naive.groupby(naive.target_date.dt.year)}
    paired = []
    for seed in (42, 123, 2026):
        row = {"seed": seed}
        entries = {r["variant"]: r["evaluation_metrics"] for r in runs if r["seed"] == seed}
        for metric in ("mae_usd", "rmse_usd"):
            row[f"B_minus_A_{metric}"] = entries["B"][metric] - entries["A"][metric]
            row[f"C_minus_B_{metric}"] = entries["C"][metric] - entries["B"][metric]
        paired.append(row)
    # These plots use training-period data; validation outcomes cannot influence feature definitions.
    raw = pd.read_json(base / "indicators.json", orient="table").loc[:"2022-12-31"]
    features = pd.read_json(base / "features.json", orient="table").loc[:"2022-12-31"]
    figures = base / "figures"
    figures.mkdir(exist_ok=False)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "font.size": 10})
    view = raw.loc["2021-01-01":]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout="constrained")
    for name in ("close", "sma20", "sma50", "ema12", "ema26"):
        axes[0].plot(view.index, view[name], label=name, linewidth=1.2)
    axes[0].set(ylabel="Split-adjusted USD", title="NVDA training period: price and smoothing indicators")
    axes[0].legend(ncol=5, fontsize=9)
    features.loc[view.index, ["sma20_distance", "sma50_distance", "ema12_distance", "ema26_distance"]].plot(ax=axes[1])
    axes[1].set(ylabel="Close / average − 1", xlabel="")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(view.index, view.rsi14, color="#7c3aed", label="RSI(14)")
    axes[2].axhline(50, color="gray", linestyle=":")
    axes[2].set(ylabel="RSI (0–100)", ylim=(0, 100))
    axes[2].legend()
    for name in ("macd", "signal", "histogram"):
        axes[3].plot(view.index, view[name], label=name)
    axes[3].set(ylabel="MACD / signal (USD)", xlabel="Session date")
    axes[3].legend(ncol=3)
    fig.savefig(figures / "technical_indicators.png", dpi=150)
    plt.close(fig)

    # Correlations are descriptive training associations, not predictive importance.
    corr = features.iloc[49:-1].corr()
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i,j]) > .65 else "black")
    ax.set_title("Unique fitting input rows: descriptive feature correlations\nNot indicator importance or causal evidence")
    fig.colorbar(im, ax=ax, shrink=.8, label="Pearson correlation")
    fig.savefig(figures / "feature_relationships.png", dpi=150)
    plt.close(fig)

    context = features.loc["2022-07-01":, ["spy_log_return", "xlk_log_return"]].copy()
    context["nvda_log_return"] = np.log(raw.close / raw.close.shift(1)).reindex(context.index)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    context.plot(ax=ax, alpha=.7, linewidth=1)
    ax.set(title="Observed training-period price returns: correlated market context", ylabel="One-session log return", xlabel="Session date")
    fig.savefig(figures / "market_context.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("mae_usd", "rmse_usd"), ("MAE", "RMSE")):
        labels = ["Naive", "ARIMA", "A", "B", "C"]
        means = [baselines["naive"][metric], baselines["arima"][metric], *[grouped[v][metric]["mean"] for v in "ABC"]]
        sds = [0, 0, *[grouped[v][metric]["std"] for v in "ABC"]]
        ax.bar(labels, means, yerr=sds, capsize=4, color=["#94a3b8", "#64748b", *COLORS.values()])
        for i, mean in enumerate(means): ax.text(i, mean + .035, f"{mean:.4f}", ha="center", fontsize=9)
        ax.set(ylabel="USD", title=title, ylim=(0, max(means) * 1.2))
    fig.suptitle("2023–2024: 502 shared targets; error bars are three-seed SD")
    fig.savefig(figures / "benchmark_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, v in zip(axes, "ABC"):
        for seed in (42, 123, 2026):
            h = pd.DataFrame(json.loads((base / f"{v}_seed_{seed}" / "history.json").read_text()))
            assert len(h) == 19 and "validation_mse" not in h
            ax.plot(h.epoch, h.training_inference_mse, label=f"Seed {seed}")
        ax.set(title=f"Model {v}", xlabel="Epoch")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Training inference MSE (standardised correction)")
    fig.suptitle("Fixed 19-epoch fits: no validation monitoring or early stopping")
    fig.savefig(figures / "training_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for v in "ABC":
        differences = []
        for seed in (42, 123, 2026):
            table = tables[f"{v}_seed_{seed}"]
            excess = np.abs(table.prediction - table.actual_close) - np.abs(naive.prediction - naive.actual_close)
            cumulative = excess.cumsum()
            differences.append(cumulative.to_numpy())
            ax.plot(table.target_date, cumulative, color=COLORS[v], alpha=.22, linewidth=.8)
        ax.plot(table.target_date, np.mean(differences, axis=0), color=COLORS[v], label=f"{v}: mean cumulative error difference")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Cumulative absolute error relative to naive (below zero is better)", ylabel="Cumulative excess absolute error (USD)", xlabel="Target date")
    ax.legend(fontsize=9)
    fig.savefig(figures / "aligned_error_comparison.png", dpi=150)
    plt.close(fig)
    compact = [{k: run[k] for k in ("variant", "seed", "status", "evaluation_metrics", "by_year",
                                   "epochs_completed", "trainable_parameters", "optimizer_steps",
                                   "reload_max_abs_difference", "elapsed_seconds")} for run in runs]
    report = {"attempts": len(runs), "failed_attempts": 0, "test_evaluated": False,
              "runs": compact, "aggregate": grouped, "annual_aggregate": annual,
              "paired_differences": paired, "baselines": baselines, "annual_baselines": annual_baselines,
              "eligibility": json.loads((base / "eligibility.json").read_text()),
              "provenance": json.loads((base / "provenance.json").read_text()),
              "figures": [str(f.relative_to(ROOT)) for f in sorted(figures.glob("*.png"))]}
    with (ROOT / "docs/feature_results_v1.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"aggregate": grouped, "annual": annual, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
