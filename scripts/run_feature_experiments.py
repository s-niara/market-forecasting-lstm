"""Execute the nine fixed Milestone 6 fits without touching Milestone 5 artifacts."""
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from market_forecasting.features import calculate_features, prepare_feature_dataset, VARIANTS
from market_forecasting.dataset import error_metrics
from market_forecasting.preprocessing import reconstruct_prices
from market_forecasting.lstm import configure_runtime, build_model, train_model, predict_standardised
from market_forecasting.calendar import audit_sessions
from run_lstm_experiments import load_stage, write_json

ROOT = Path(__file__).resolve().parents[1]


def execute_feature_run(data, settings, variant, seed, provenance, output):
    """Preserve one attempted fixed-epoch fit and its success or failure evidence."""
    output.mkdir(exist_ok=False)
    record = dict(variant=variant, seed=seed, status="started", settings=settings,
                  scaling=data.provenance, provenance=provenance, test_evaluated=False,
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "summary.json", record)
    try:
        fit, evaluate = data.fit_mask, ~data.fit_mask
        model = build_model(settings, seed, lookback=30, features=data.X.shape[2])
        # No validation argument: every variant trains for exactly the same 19 epochs.
        training = train_model(model, data.X[fit], data.y[fit], settings, output)
        correction = predict_standardised(model, data.X[evaluate])
        loaded = tf.keras.models.load_model(output / "model.keras")
        repeated = predict_standardised(loaded, data.X[evaluate])
        np.testing.assert_allclose(correction, repeated, rtol=0, atol=1e-7)
        prices = reconstruct_prices(data.origins[evaluate], correction, data.correction_scaler)
        table = data.metadata.loc[evaluate, ["origin_date", "target_date"]].reset_index(drop=True)
        table["origin_close"], table["actual_close"] = data.origins[evaluate], data.targets[evaluate]
        table["predicted_correction"] = data.correction_scaler.inverse_transform(correction)
        table["prediction"] = prices
        table.to_json(output / "predictions.json", orient="records", date_format="iso")
        record.update(training, status="complete", evaluation_metrics=error_metrics(table.actual_close, prices),
                      naive_metrics=error_metrics(table.actual_close, table.origin_close),
                      reload_max_abs_difference=float(np.max(np.abs(correction - repeated))),
                      by_year={str(y): error_metrics(part.actual_close, part.prediction)
                               for y, part in table.groupby(table.target_date.dt.year)})
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_json(output / "summary.json", record)
    print(json.dumps({k: record.get(k) for k in ("variant", "seed", "status", "evaluation_metrics", "error")}), flush=True)
    return record


def main():
    """Validate the locked protocol and eligibility before starting any training."""
    configure_runtime()
    config = json.loads((ROOT / "configs/nvda.json").read_text())
    path = ROOT / "configs/feature_experiment_v1.json"
    settings = json.loads(path.read_text())
    selected = json.loads((ROOT / "configs/lstm_selected_v1.json").read_text())
    for key in ("units", "dropout", "batch_size", "learning_rate"):
        if settings[key] != selected["configuration"][key]:
            raise ValueError("Milestone 5 hyperparameters must remain fixed")
    if settings["epochs"] != selected["epochs"] or settings["seeds"] != selected["seeds"] or settings["lookback"] != selected["lookback"]:
        raise ValueError("Epoch, seed or lookback protocol changed")
    if settings["variants"] != VARIANTS or settings["early_stopping"] or settings["warmup_rows"] != 49:
        raise ValueError("Feature or stopping protocol changed")
    # These definitions are fixed in the implementation; metadata must not imply a different calculation.
    fixed_specification = {"sma": [20, 50], "ema": [12, 26], "rsi": 14, "macd": [12, 26, 9],
                           "optimizer": "Adam", "loss": "standardised_correction_mse",
                           "formulation": "residual_dollar_correction"}
    for key, expected in fixed_specification.items():
        if settings.get(key) != expected:
            raise ValueError(f"Fixed feature protocol mismatch for {key}")
    series, hashes = {}, {}
    for ticker in ("NVDA", "SPY", "XLK"):
        series[ticker], hashes[ticker] = load_stage(dict(config, target_ticker=ticker), config["validation_end"])
        # NYSE is the existing documented session proxy for SPY and XLK.
        calendar = "NASDAQ" if ticker == "NVDA" else "NYSE"
        coverage = audit_sessions(series[ticker].index, config["start"], config["validation_end"], calendar)
        if coverage["missing"] or coverage["unexpected"]:
            raise ValueError("Context calendar mismatch")
    raw, features = calculate_features(series, config)
    prepared = {v: prepare_feature_dataset(features, config, v) for v in VARIANTS}
    baseline = pd.read_json(ROOT / "artifacts/naive_v1/validation_predictions.json")
    expected_dates = pd.to_datetime(baseline.target_date)
    for data in prepared.values():
        pd.testing.assert_frame_equal(data.metadata, prepared["A"].metadata)
        np.testing.assert_array_equal(data.y, prepared["A"].y)
        actual_dates = data.metadata.loc[~data.fit_mask, "target_date"].reset_index(drop=True)
        # Snapshot and JSON readers use different datetime resolutions; compare dates in nanoseconds.
        pd.testing.assert_series_equal(actual_dates.astype("datetime64[ns]"),
                                       expected_dates.astype("datetime64[ns]"), check_names=False)
        if len(actual_dates) != 502:
            raise ValueError("Expected all 502 validation dates")
    provenance = {"config": config, "feature_protocol": settings, "snapshot_sha256": hashes,
                  "feature_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "selected_config_sha256": hashlib.sha256((ROOT / "configs/lstm_selected_v1.json").read_bytes()).hexdigest(),
                  "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                    for f in [Path(__file__).resolve(), ROOT / "scripts/run_lstm_experiments.py",
                                              *sorted((ROOT / "src/market_forecasting").glob("*.py"))]},
                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                  "device": "CPU", "threads_intra": 2, "threads_inter": 1,
                                  "deterministic_ops": True,
                                  **{name: version(name) for name in ("tensorflow", "keras", "numpy", "pandas")}}}
    output = ROOT / "artifacts/features_v1"
    # Exclusive creation makes a second execution fail before any model is fitted.
    output.mkdir(exist_ok=False)
    write_json(output / "provenance.json", provenance)
    write_json(output / "eligibility.json", {v: d.provenance for v, d in prepared.items()})
    raw.to_json(output / "indicators.json", orient="table", date_format="iso")
    features.to_json(output / "features.json", orient="table", date_format="iso")
    runs = []
    for variant, data in prepared.items():
        for seed in settings["seeds"]:
            runs.append(execute_feature_run(data, settings, variant, seed, provenance,
                                             output / f"{variant}_seed_{seed}"))
            write_json(output / "runs.json", runs)
    if any(run["status"] != "complete" for run in runs):
        raise RuntimeError("Failures retained; no replacement fits are authorised")


if __name__ == "__main__":
    main()
