"""Summarise the fixed LSTM experiment and render its recorded learning curves."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Read completed runs without fitting or selecting any further model."""
    base = ROOT / "artifacts/lstm_v1"
    selection = json.loads((base / "selection/selection.json").read_text())
    internal = json.loads((base / "selection/runs.json").read_text())
    final = json.loads((base / "final/runs.json").read_text())
    if len(internal) != 12 or len(final) != 3:
        raise ValueError("The report requires the approved 12 + 3 recorded attempts")
    if any(run["status"] != "complete" for run in internal + final):
        raise ValueError("Inspect recorded failures before preparing a complete comparison")
    figures = base / "figures"
    figures.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})
    histories = {}
    for run in internal:
        key = f"{run['configuration_id']}_seed_{run['seed']}"
        histories[key] = pd.DataFrame(json.loads((base / "selection" / key / "history.json").read_text()))
    for zoom in [False, True]:
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), layout="constrained")
        for ax, run in zip(axes.flat, internal):
            h = histories[f"{run['configuration_id']}_seed_{run['seed']}"]
            if not zoom:
                ax.plot(h.epoch, h.training_loss, color="#94a3b8", label="Training (minibatch)")
                ax.plot(h.epoch, h.training_inference_mse, color="#2563eb", label="Training (inference)")
            ax.plot(h.epoch, h.validation_mse, color="#d97706", label="Internal validation")
            ax.axvline(run["best_epoch"], color="#16a34a", linestyle="--", label="Best epoch")
            ax.axvline(run["epochs_completed"], color="#dc2626", linestyle=":", label="Stopping epoch")
            ax.set_title(f"{run['configuration_id']} / seed {run['seed']} — best {run['best_epoch']}, stop {run['epochs_completed']}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Standardised correction MSE")
            if not zoom:
                ax.set_yscale("log")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        fig.suptitle("Internal-validation detail (individual y scales)" if zoom else
                     "LSTM selection: training and internal-validation loss (log y scale)", fontsize=15)
        fig.savefig(figures / ("validation_curves.png" if zoom else "selection_curves.png"), dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for run in final:
        name = f"{run['configuration_id']}_seed_{run['seed']}"
        h = pd.DataFrame(json.loads((base / "final" / name / "history.json").read_text()))
        ax.plot(h.epoch, h.training_inference_mse, marker="o", label=f"Seed {run['seed']}")
    ax.set(xlabel="Epoch", ylabel="Training MSE (standardised correction)",
           title="Full-development fits: fixed epoch budget, no external early stopping")
    ax.legend()
    fig.savefig(figures / "final_training_curves.png", dpi=160)
    plt.close(fig)

    naive = json.loads((ROOT / "artifacts/naive_v1/summary.json").read_text())["validation_metrics"]
    arima = json.loads((ROOT / "artifacts/arima_v1/summary.json").read_text())["validation_metrics"]["arima"]
    expected_dates = pd.to_datetime(pd.read_json(ROOT / "artifacts/naive_v1/validation_predictions.json").target_date)
    for run in final:
        table = pd.read_json(base / "final" / f"{run['configuration_id']}_seed_{run['seed']}" / "predictions.json")
        assert pd.to_datetime(table.target_date).equals(expected_dates)
        assert run["naive_metrics"] == naive
    aggregate = {key: {"mean": float(np.mean([run["evaluation_metrics"][key] for run in final])),
                       "std": float(np.std([run["evaluation_metrics"][key] for run in final], ddof=1))}
                 for key in ["mae_usd", "rmse_usd"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(axes, ["mae_usd", "rmse_usd"], ["MAE", "RMSE"]):
        values = [naive[metric], arima[metric], aggregate[metric]["mean"]]
        ax.bar(["Naive", "ARIMA", "LSTM (mean)"], values,
               yerr=[0, 0, aggregate[metric]["std"]], capsize=5,
               color=["#94a3b8", "#64748b", "#2563eb"])
        for i, value in enumerate(values): ax.text(i, value + .04, f"{value:.4f}", ha="center")
        ax.set(ylabel="USD", title=title, ylim=(0, max(values) * 1.18))
    fig.suptitle("2023–2024: identical 502 target dates; LSTM error bars are seed SD")
    fig.savefig(figures / "benchmark_comparison.png", dpi=160)
    plt.close(fig)

    curve_summary = []
    for run in internal:
        h = histories[f"{run['configuration_id']}_seed_{run['seed']}"]
        curve_summary.append({"configuration_id": run["configuration_id"], "seed": run["seed"],
                              "first_train_mse": float(h.training_inference_mse.iloc[0]),
                              "last_train_mse": float(h.training_inference_mse.iloc[-1]),
                              "first_val_mse": float(h.validation_mse.iloc[0]),
                              "best_val_mse": run["best_validation_mse"],
                              "last_val_mse": float(h.validation_mse.iloc[-1])})
    compact = lambda run: {k: run[k] for k in ["configuration_id", "seed", "status", "epochs_completed",
                       "best_epoch", "stopped_early", "reached_epoch_cap", "evaluation_metrics", "naive_metrics",
                       "elapsed_seconds", "optimizer_steps", "trainable_parameters", "by_year",
                       "reload_max_abs_difference"]}
    report = {"selection": selection, "internal_runs": [compact(run) for run in internal],
              "final_runs": [compact(run) for run in final], "external_aggregate": aggregate,
              "baselines": {"naive": naive, "arima": arima}, "curve_summary": curve_summary,
              "internal_scaling": internal[0]["scaling"], "final_scaling": final[0]["scaling"],
              "provenance": internal[0]["provenance"], "attempts": 15, "failed_attempts": 0,
              "test_evaluated": False}
    (ROOT / "docs/lstm_results_v1.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selection": selection, "external_aggregate": aggregate,
                      "curves": curve_summary, "final_runs": report["final_runs"]}, indent=2))


if __name__ == "__main__":
    main()
