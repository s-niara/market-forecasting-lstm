"""Run only the approved residual-LSTM selection or final-development phase."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import traceback

import numpy as np
import pandas as pd
from market_forecasting.calendar import audit_sessions
from market_forecasting.data import load_yahoo_snapshot, snapshot_provenance
from market_forecasting.dataset import error_metrics
from market_forecasting.preprocessing import prepare_residual_data, reconstruct_prices
from market_forecasting.lstm import (configure_runtime, build_model, train_model,
                                    predict_standardised, choose_configuration)
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Write finite JSON records with stable formatting."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_stage(config, end):
    """Verify and load a bounded development interval without decoding later prices."""
    if not pd.Timestamp(config["start"]) <= pd.Timestamp(end) <= pd.Timestamp(config["validation_end"]):
        raise ValueError("Stage end must remain within the configured development interval")
    snapshot = ROOT / "data/raw" / f"{config['target_ticker']}_yahoo_snapshot.json"
    provenance = snapshot_provenance(snapshot, ROOT / "docs/data_verification.json", config["target_ticker"])
    close = load_yahoo_snapshot(snapshot, start=config["start"], end=end)["close"].astype(float)
    # Preserve the original unnamed Series and session index for both experiment runners.
    close = close.rename(None).rename_axis(None)
    coverage = audit_sessions(close.index, config["start"], end, "NASDAQ")
    if coverage["missing"] or coverage["unexpected"] or coverage["duplicate_dates"] or not coverage["ordered"]:
        raise ValueError("Development calendar coverage failed")
    return close, provenance["sha256"]


def execute_run(prepared, cfg, seed, phase, search, provenance, output):
    """Record each attempted fit, including any failure, in a unique run directory."""
    output.mkdir(exist_ok=False)
    record = dict(status="started", phase=phase, configuration_id=cfg["id"], seed=seed,
                  settings=cfg, provenance=provenance, scaling=prepared.provenance,
                  started_at_utc=datetime.now(timezone.utc).isoformat(), test_evaluated=False)
    write_json(output / "summary.json", record)
    try:
        model = build_model(cfg, seed, search["lookback"])
        fit, evaluation = prepared.fit_mask, ~prepared.fit_mask
        settings = dict(cfg, patience=search["patience"])
        val = (prepared.X[evaluation], prepared.y[evaluation]) if phase == "selection" else None
        training = train_model(model, prepared.X[fit], prepared.y[fit], settings, output, val)
        standardised = predict_standardised(model, prepared.X[evaluation])
        # Reloading checks the saved checkpoint without an additional training run.
        reloaded = tf.keras.models.load_model(output / "model.keras")
        repeated = predict_standardised(reloaded, prepared.X[evaluation])
        np.testing.assert_allclose(standardised, repeated, rtol=0, atol=1e-7)
        data = prepared.dataset
        origins = data.X[evaluation, -1, 0]
        prices = reconstruct_prices(origins, standardised, prepared.correction_scaler)
        table = data.metadata.loc[evaluation, ["origin_date", "target_date"]].reset_index(drop=True)
        table["origin_close"] = origins
        table["actual_close"] = data.y[evaluation]
        table["predicted_correction"] = prepared.correction_scaler.inverse_transform(standardised)
        table["prediction"] = prices
        table.to_json(output / "predictions.json", orient="records", date_format="iso")
        metrics = error_metrics(data.y[evaluation], prices)
        record.update(training, status="complete", evaluation_metrics=metrics,
                      naive_metrics=error_metrics(data.y[evaluation], origins),
                      reload_max_abs_difference=float(np.max(np.abs(standardised - repeated))),
                      by_year={str(year): error_metrics(part.actual_close, part.prediction)
                               for year, part in table.groupby(table.target_date.dt.year)})
        # Verify restored checkpoint loss, not just the stopping epoch's weights.
        if phase == "selection":
            restored_mse = float(np.mean((standardised - prepared.y[evaluation]) ** 2))
            np.testing.assert_allclose(restored_mse, training["best_validation_mse"], rtol=1e-6, atol=1e-8)
            record["restored_validation_mse"] = restored_mse
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_json(output / "summary.json", record)
    print(json.dumps({k: record.get(k) for k in ["phase", "configuration_id", "seed", "status",
                                               "best_epoch", "epochs_completed", "evaluation_metrics", "error"]}), flush=True)
    return record


def main():
    """Keep internal selection separate from external-validation scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["selection", "final"])
    args = parser.parse_args()
    configure_runtime()
    config = json.loads((ROOT / "configs/nvda.json").read_text())
    search_path = ROOT / "configs/lstm_search_v1.json"
    search = json.loads(search_path.read_text())
    phase_dir = ROOT / "artifacts/lstm_v1" / args.phase
    # An existing phase is never resumed or replaced silently, preserving the run budget.
    phase_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "search_sha256": hashlib.sha256(search_path.read_bytes()).hexdigest(),
        "forecast_config": config, "search": search,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), *sorted((ROOT / "src/market_forecasting").glob("*.py"))]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "device": "CPU", "threads_intra": 2, "threads_inter": 1, "deterministic_ops": True,
                        **{p: version(p) for p in ["tensorflow", "keras", "numpy", "pandas"]}}}
    if args.phase == "selection":
        end, fit_end = config["train_end"], search["internal_fit_end"]
        configs = [dict(c, epochs=search["max_epochs"]) for c in search["configurations"]]
    else:
        selected = json.loads((ROOT / "configs/lstm_selected_v1.json").read_text())
        if selected["search_sha256"] != provenance["search_sha256"]:
            raise ValueError("Search configuration changed since selection")
        end, fit_end = config["validation_end"], config["train_end"]
        configs = [dict(selected["configuration"], epochs=selected["epochs"])]
    close, sha = load_stage(config, end)
    provenance["snapshot_sha256"] = sha
    prepared = prepare_residual_data(close, config, fit_end, end)
    runs = []
    for cfg in configs:
        for seed in search["seeds"]:
            runs.append(execute_run(prepared, cfg, seed, args.phase, search, provenance,
                                    phase_dir / f"{cfg['id']}_seed_{seed}"))
            write_json(phase_dir / "runs.json", runs)
    if args.phase == "selection":
        selection = choose_configuration(runs, search["configurations"], search["seeds"])
        selection.update(configuration=next(c for c in search["configurations"] if c["id"] == selection["selected_id"]),
                         search_sha256=provenance["search_sha256"], seeds=search["seeds"],
                         formulation="standardised_dollar_correction_plus_origin_close", lookback=search["lookback"],
                         optimizer="Adam", loss="standardised_correction_mse", scaling="training_only_standardisation",
                         protocol=search)
        selected_path = ROOT / "configs/lstm_selected_v1.json"
        with selected_path.open("x") as f:
            json.dump(selection, f, indent=2, allow_nan=False)
        write_json(phase_dir / "selection.json", selection)
        print(json.dumps(selection), flush=True)
    elif any(run["status"] != "complete" for run in runs):
        raise RuntimeError("Final-development failures recorded; no replacement runs performed")


if __name__ == "__main__":
    main()
