"""Fit, independently verify and freeze pre-test models; test evaluation stays sealed."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from market_forecasting.arima import fit_fixed_order, FrozenArima
from market_forecasting.calendar import audit_sessions
from market_forecasting.features import (VARIANTS, calculate_features, prepare_final_feature_dataset,
                                         prepare_origin_input, validate_feature_preprocessing)
from market_forecasting.lstm import (configure_runtime, build_model, train_model,
                                     load_frozen_model, frozen_forecast)
from market_forecasting.preprocessing import Standardisation
from run_lstm_experiments import load_stage, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/final_training_v1.json"
SEALED_BOUNDARY = pd.Timestamp("2024-12-31")


def now():
    """Record an unambiguous UTC event time."""
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    """Hash file bytes without interpreting market-price payloads."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    """Read a configuration or experiment record."""
    return json.loads(Path(path).read_text())


def require(condition, message):
    """Stop on a violated protocol invariant, including under optimised Python."""
    if not condition:
        raise ValueError(message)


def guard_protocol(protocol, config, selected, feature, arima):
    """Reject departures from the approved protocol before opening market data."""
    d, c, a, l = (protocol[k] for k in ("data", "common_eligibility", "arima", "lstm"))
    require(d["start"] == config["start"] == "2015-01-01"
            and d["fit_end"] == config["validation_end"] == "2024-12-31"
            and config["train_end"] == "2022-12-31" and config["test_end"] == "2025-12-31",
            "Chronological boundaries differ from the approved pre-test protocol")
    require(d["target_ticker"] == config["target_ticker"] == "NVDA"
            and d["context_tickers"] == config["context_tickers"] == ["SPY", "XLK"] and d["target_column"] == "close"
            and d["price_convention"] == config["price_convention"] == "split_adjusted_close"
            and d["forecast_horizon_trading_days"] == config["forecast_horizon_trading_days"] == 1,
            "Target, source instruments or price convention changed")
    require(d["bounded_quote_decoding_required"] and d["snapshot_policy"] == "existing_frozen_snapshots_only",
            "Bounded frozen-snapshot loading is mandatory")
    require(d["calendars"] == {"NVDA": "NASDAQ", "SPY": "NYSE", "XLK": "NYSE"},
            "Instrument calendars changed")
    require(d["expected_sessions"] == 2516 and c["warmup_rows"] == feature["warmup_rows"] == 49
            and c["expected_fit_examples"] == 2437 and c["expected_unique_input_rows"] == 2466
            and c["include_last_target_in_input_scaler"] is False and c["lookback"] == 30
            and c["input_scaling"] == "unique_fitting_input_observations_population_standard_deviation"
            and c["target_scaling"] == "fitting_dollar_corrections_population_standard_deviation"
            and c["constant_series_scale"] == 1.0,
            "Final fitting eligibility convention changed")
    fixed = {"units": 16, "dropout": 0.2, "learning_rate": 0.001, "batch_size": 32}
    for key, value in fixed.items():
        require(l[key] == feature[key] == selected["configuration"][key] == value,
                f"Frozen LSTM setting changed: {key}")
    for key, value in {"lookback": 30, "epochs": 19, "seeds": [42, 123, 2026],
                       "optimizer": "Adam", "loss": "standardised_correction_mse"}.items():
        require(l[key] == feature[key] == selected[key] == value, f"Frozen training protocol changed: {key}")
    require(config["initial_sequence_length"] == 30 and l["feature_names"] == feature["variants"] == VARIANTS
            and l["formulation"] == feature["formulation"] == "residual_dollar_correction"
            and not l["early_stopping"] and not l["internal_validation"] and not feature["early_stopping"]
            and l["fresh_initialisation"] and l["batch_order"] == "chronological"
            and l["expected_fits"] == 9 and l["test_weights"] == l["test_scalers"] == "fixed"
            and l["reconstruction"] == "origin_close_plus_inverse_standardised_correction", "Final LSTM formulation or comparison changed")
    for key, value in {"sma": [20, 50], "ema": [12, 26], "rsi": 14, "macd": [12, 26, 9]}.items():
        require(l["feature_specification"][key] == feature[key] == value, "Indicator specification changed")
    require(a["order"] == [2, 1, 2] and a["trend"] == arima["trend"] == "n"
            and a["maxiter"] == arima["maxiter"] == 500 and a["enforce_stationarity"]
            and a["enforce_invertibility"] and not a["order_search"] and arima["refit"] is False,
            "Frozen ARIMA specification changed")
    e = protocol["evaluation_protocol"]
    require(e["test_sealed"] and not e["test_access_authorised"] and not e["execution_authorised"],
            "Independent testing is not authorised by the final-training operation")
    require(l["expected_parameters"] == {v: 4 * 16 * (len(names) + 17) + 17 for v, names in VARIANTS.items()}
            and l["expected_optimizer_steps_per_epoch"] == 77 and l["expected_optimizer_steps"] == 1463,
            "Architecture or optimizer-step metadata changed")
    comparison = protocol["comparison"]
    require(comparison["families"] == ["naive", "ARIMA", "A", "B", "C"]
            and comparison["primary_lstm"] == "A" and comparison["benchmarks"] == ["naive", "ARIMA"]
            and comparison["feature_ablations"] == [["B", "A"], ["C", "B"]]
            and comparison["paired_error_comparisons"] == [["A", "naive"], ["ARIMA", "naive"], ["B", "A"], ["C", "B"]]
            and not comparison["prediction_ensemble"] and not comparison["milestone_5_separate_final_refit"]
            and comparison["aggregate"] == "arithmetic_mean_of_individual_seed_metrics",
            "Prespecified model comparison changed")
    require(e["test_start"] == "2025-01-01" and e["test_end"] == "2025-12-31"
            and e["expected_calendar_sessions"] == 250 and e["primary_metric"] == "mae_usd"
            and e["secondary_metric"] == "rmse_usd" and e["representative_seed"] == 42
            and e["retain_pretest_indicator_history"] and e["retain_pretest_input_context"]
            and e["flow"] == "forecast_then_observe_then_update_history"
            and e["illustrative_examples"] == {"first_test_forecast": True,
                "largest_naive_absolute_error_dates": 3, "same_dates_for_all_models": True},
            "Future evaluation boundaries, timing, metrics or examples changed")
    b = e["bootstrap"]
    require(b["method"] == "paired_circular_block_bootstrap" and b["metric"] == "mae_difference"
            and b["primary_block_length"] == 10 and b["sensitivity_block_lengths"] == [5, 20]
            and b["resamples"] == 5000 and b["seed"] == 314159 and b["interval_level"] == .95
            and b["resample_dates_jointly_across_models_and_seeds"], "Future uncertainty protocol changed")
    require(protocol["return_target_experiment"] == "deferred_not_authorised", "Return target remains deferred")
    require(protocol["artifacts"]["directory"] == "artifacts/final_training_v1"
            and not protocol["artifacts"]["overwrite_existing"]
            and protocol["artifacts"]["independent_reload_verification_required"]
            and protocol["artifacts"]["freeze_only_after_all_required_verification_passes"], "Final artifact destination changed")


def approved_configuration():
    """Validate the final specification against the established development files."""
    protocol = read_json(PROTOCOL_PATH)
    config = read_json(ROOT / "configs/nvda.json")
    guard_protocol(protocol, config, read_json(ROOT / "configs/lstm_selected_v1.json"),
                   read_json(ROOT / "configs/feature_experiment_v1.json"),
                   read_json(ROOT / "configs/arima.json"))
    return protocol, config


def prepare_final(protocol, config):
    """Load only pre-test quotes and derive the identical final fitting samples."""
    series, hashes = {}, {}
    for ticker, calendar in protocol["data"]["calendars"].items():
        series[ticker], hashes[ticker] = load_stage(dict(config, target_ticker=ticker), "2024-12-31")
        audit = audit_sessions(series[ticker].index, config["start"], "2024-12-31", calendar)
        require(len(series[ticker]) == protocol["data"]["expected_sessions"]
                and not audit["missing"] and not audit["unexpected"], "Pre-test calendar coverage mismatch")
    _, features = calculate_features(series, config)
    prepared = {v: prepare_final_feature_dataset(features, config, v, "2024-12-31") for v in VARIANTS}
    expected = protocol["common_eligibility"]
    for v, data in prepared.items():
        p = data.provenance
        for key in ("first_feature_date", "first_target_date", "last_fitting_input_date", "last_fitting_target_date"):
            require(p[key] == expected["expected_" + key], f"Unexpected {key}")
        require(p["fit_examples"] == expected["expected_fit_examples"]
                and p["input_unique_rows"] == expected["expected_unique_input_rows"]
                and p["validation_examples"] == 0 and data.fit_mask.all(), "Unexpected final fitting counts")
        require(data.X.shape == (p["fit_examples"], 30, len(VARIANTS[v]))
                and np.isfinite(data.X).all() and np.isfinite(data.y).all(), "Invalid final fitting tensor")
        pd.testing.assert_frame_equal(data.metadata, prepared["A"].metadata)
        np.testing.assert_array_equal(data.y, prepared["A"].y)
        np.testing.assert_array_equal(data.targets, prepared["A"].targets)
        np.testing.assert_array_equal(data.origins, prepared["A"].origins)
        for name in set(VARIANTS[v]) & set(VARIANTS["B"]):
            require(p["input_scalers"][name] == prepared["B"].provenance["input_scalers"][name],
                    "Shared feature scaling differs between variants")
    return series, hashes, prepared


def verify_hashes(hashes):
    """Reject changed or missing files before loading their contents."""
    for relative, expected in hashes.items():
        path = ROOT / relative
        require(path.is_file() and sha256(path) == expected, f"Integrity mismatch: {relative}")


def protected_hashes():
    """Identify immutable development evidence and user-owned housekeeping files."""
    files = [ROOT / "START_HERE.md", ROOT / "docs/PLAN.md", *sorted((ROOT / "docs").glob("*.json"))]
    for directory in ("notebooks", "data/raw", "artifacts/naive_v1", "artifacts/arima_v1",
                      "artifacts/lstm_v1", "artifacts/features_v1"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.is_file()}


def source_snapshot(output):
    """Archive the actual uncommitted implementation as well as its Git revision."""
    files = []
    for directory in ("src", "scripts", "tests", "configs"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json"))
    files.extend(p for p in ROOT.glob("*") if p.is_file() and
                 (p.name.startswith("requirements") or p.name in ("pyproject.toml", "uv.lock")))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in hashes:
            archive.write(ROOT / relative, relative)
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=ROOT, text=True).strip()
    return {"revision": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short"),
            "source_sha256": hashes, "source_archive": str((output / "source.zip").relative_to(ROOT))}


def arima_checks(model):
    """Capture fitted state and in-sample predictions without forecasting beyond fitting."""
    indices = [0, int(model.nobs) - 2, int(model.nobs) - 1]
    return {"nobs": int(model.nobs), "parameters": np.asarray(model.params).tolist(),
            "filtered_state": model.filter_results.filtered_state[:, -1].tolist(),
            "filtered_state_cov": model.filter_results.filtered_state_cov[:, :, -1].tolist(),
            "predicted_state": model.filter_results.predicted_state[:, -1].tolist(),
            "predicted_state_cov": model.filter_results.predicted_state_cov[:, :, -1].tolist(),
            "in_sample_indices": indices,
            "in_sample_predictions": [float(np.asarray(model.predict(start=i, end=i))[0]) for i in indices]}


def pretest_forecast(series, config, origin, target, bundles, arima_cursor):
    """Issue aligned pre-test forecasts without accepting any target observation."""
    origin, target = pd.Timestamp(origin), pd.Timestamp(target)
    require(target <= SEALED_BOUNDARY and origin < target, "Independent test forecasts remain sealed")
    audit = audit_sessions(pd.DatetimeIndex([origin, target]), origin, target, "NASDAQ")
    require(not audit["missing"] and not audit["unexpected"] and audit["expected_sessions"] == 2,
            "Forecast dates must be consecutive exchange sessions")
    predictions = {}
    for identity, bundle in bundles.items():
        X, close, _ = prepare_origin_input(series, config, bundle["variant"], origin, bundle["preprocessing"])
        result = frozen_forecast(bundle["model"], X, np.array([close]), bundle["correction_scaler"])
        predictions[identity] = float(result["prediction"][0])
        predictions.setdefault("naive", close)
        require(predictions["naive"] == close, "Forecast origins are not aligned")
    require(bool(bundles), "At least one frozen LSTM bundle is required")
    predictions["ARIMA"] = arima_cursor.forecast(origin, target)
    return {"origin_date": str(origin.date()), "target_date": str(target.date()), "predictions": predictions}


def artifact_hashes(output):
    """Inventory immutable payloads, excluding the manifest's self-reference."""
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}


def verify(output):
    """Independently reload every final artifact using pre-test observations only."""
    manifest = read_json(output / "manifest.json")
    require(manifest["status"] in ("awaiting_verification", "frozen"), "Incomplete final fitting run")
    verify_hashes(manifest["artifacts_sha256"])
    verify_hashes(manifest["protected_sha256"])
    verify_hashes(manifest["source"]["source_sha256"])
    protocol, config = approved_configuration()
    require(protocol == manifest["protocol"], "Final protocol differs from the saved manifest")
    series, hashes, prepared = prepare_final(protocol, config)
    require(hashes == manifest["snapshot_sha256"], "Snapshot identity mismatch")
    settings = protocol["lstm"]
    bundles, checks = {}, []
    expected_ids = [f"{v}_seed_{seed}" for v in VARIANTS for seed in settings["seeds"]]
    require([run["identity"] for run in manifest["runs"]] == expected_ids, "Final run identities mismatch")
    for record in manifest["runs"]:
        v, seed, identity = record["variant"], record["seed"], record["identity"]
        require(identity == f"{v}_seed_{seed}" and seed in settings["seeds"], "Model identity metadata mismatch")
        directory = output / identity
        summary = read_json(directory / "summary.json")
        require(record == summary and summary["status"] == "complete", "Final run record mismatch")
        preprocessing = read_json(output / f"preprocessing_{v}.json")
        validate_feature_preprocessing(preprocessing, v)
        require(preprocessing == prepared[v].provenance, "Frozen scaler reconstruction mismatch")
        model = load_frozen_model(directory / "model.keras", VARIANTS[v], settings)
        require(model.count_params() == settings["expected_parameters"][v]
                and model.layers[1].get_config()["seed"] == seed, "Parameter count or Dropout seed mismatch")
        history = read_json(directory / "history.json")
        require(len(history) == 19 and [r["epoch"] for r in history] == list(range(1, 20))
                and all(r["optimizer_steps"] == 77 * r["epoch"] for r in history)
                and all(np.isfinite(r["training_loss"]) and np.isfinite(r["training_inference_mse"])
                        and "validation_mse" not in r for r in history), "Invalid fixed-epoch history")
        # Saved optimizer state is checked separately; inference never attaches an optimizer.
        import h5py
        import io
        with zipfile.ZipFile(directory / "model.keras") as archive:
            with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
                require(int(weights["optimizer/vars/0"][()]) == 1463, "Saved optimizer step count mismatch")
        data = prepared[v]
        rows = summary["check_indices"]
        require(rows == [0, len(data.X) - 2, len(data.X) - 1]
                and summary["check_target_dates"] == [str(data.metadata.target_date.iloc[i].date()) for i in rows],
                "Pre-test check indices or target dates changed")
        result = frozen_forecast(model, data.X[rows], data.origins[rows], data.correction_scaler)
        repeated = frozen_forecast(model, data.X[rows], data.origins[rows], data.correction_scaler)
        maximum = 0.0
        for key in result:
            np.testing.assert_allclose(result[key], summary["pretest_inference"][key], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(result[key], repeated[key])
            maximum = max(maximum, float(np.max(np.abs(result[key] - summary["pretest_inference"][key]))))
        # Independent origin transformation must reproduce windows constructed for fitting.
        for row in rows:
            meta = data.metadata.iloc[row]
            X, close, _ = prepare_origin_input(series, config, v, meta.origin_date, preprocessing)
            np.testing.assert_array_equal(X[0], data.X[row])
            require(close == data.origins[row], "Origin Close reconstruction mismatch")
        bundles[identity] = {"variant": v, "model": model, "preprocessing": preprocessing,
                             "correction_scaler": Standardisation(**preprocessing["correction_scaler"])}
        checks.append({"identity": identity, "seed": seed, "status": "passed",
                       "reload_max_absolute_difference": maximum, "features": len(VARIANTS[v]),
                       "parameters": model.count_params(), "epochs": 19, "optimizer_steps": 1463})
    fitted = ARIMAResults.load(output / "arima.pkl")
    require(arima_checks(fitted) == manifest["arima"]["checks"], "Restored ARIMA fitted state differs")
    require(tuple(fitted.model.order) == (2, 1, 2) and fitted.model.trend == "n"
            and fitted.model.enforce_stationarity and fitted.model.enforce_invertibility
            and fitted.nobs == 2516, "Restored ARIMA specification mismatch")
    # This is a machinery check, not accuracy evidence: coefficients were fitted through Dec 31.
    # Refilter a shorter prefix with those fixed coefficients; do not fit or forecast into 2025.
    nvda = series["NVDA"]
    prefix = fitted.apply(nvda.iloc[:-2].to_numpy(), refit=False)
    cursor = FrozenArima(prefix, nvda.index[-3])
    sequence = []
    for target in nvda.index[-2:]:
        issued = pretest_forecast(series, config, cursor.last_observed_date, target, bundles, cursor)
        require(len(issued["predictions"]) == 11, "All five families and nine seeds must share each date")
        cursor.observe(target, float(nvda.loc[target]))
        sequence.append({"origin_date": issued["origin_date"], "target_date": issued["target_date"],
                         "forecast_count": len(issued["predictions"]), "observed_after_forecast": True})
    verify_hashes(manifest["protected_sha256"])
    verify_hashes(manifest["artifacts_sha256"])
    return {"status": "passed", "verified_at_utc": now(), "independent_process": True,
            "process_id": __import__("os").getpid(), "lstm_checks": checks,
            "arima_restored_state_exact": True, "arima_fixed_parameter_continuation": True,
            "pretest_sequence": sequence, "pretest_checks_are_accuracy_evidence": False,
            "frozen_weights_scalers": True, "historical_feature_continuity": True,
            "development_artifacts_unchanged": True, "test_observations_decoded": False,
            "test_forecasts_generated": False}


def fit_and_freeze():
    """Run each approved fit once and freeze only after a fresh-process verification."""
    protocol, config = approved_configuration()
    output = ROOT / protocol["artifacts"]["directory"]
    # Refuse all existing destinations, including failures; never silently resume or replace a fit.
    require(not output.exists(), "Final artifact destination already exists; no automatic refitting is allowed")
    series, hashes, prepared = prepare_final(protocol, config)
    protected = protected_hashes()
    output.mkdir(exist_ok=False)
    manifest = {"version": 1, "status": "started", "started_at_utc": now(), "protocol": protocol,
                "snapshot_sha256": hashes, "protected_sha256": protected,
                "source": source_snapshot(output), "runs": [],
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "CPU", "threads_intra": 2, "threads_inter": 1, "deterministic_ops": True,
                    **{name: version(name) for name in ("tensorflow", "keras", "numpy", "pandas",
                                                       "statsmodels", "scipy", "pandas_market_calendars", "h5py")}},
                "observations": {"first_session": str(series["NVDA"].index[0].date()),
                                 "last_session": str(series["NVDA"].index[-1].date()), "sessions": len(series["NVDA"])},
                "test_observations_decoded": False, "test_forecasts_generated": False}
    write_json(output / "manifest.json", manifest)
    try:
        started = now()
        manifest["arima"] = {"identity": "ARIMA_2_1_2", "status": "started", "started_at_utc": started}
        write_json(output / "manifest.json", manifest)
        fitted, evidence = fit_fixed_order(series["NVDA"].to_numpy(), order=(2, 1, 2), maxiter=500)
        fitted.save(output / "arima.pkl")
        manifest["arima"] = {"identity": "ARIMA_2_1_2", "status": "complete",
                              "started_at_utc": started, "completed_at_utc": now(),
                              "fit": evidence, "checks": arima_checks(fitted),
                              "artifact": str((output / "arima.pkl").relative_to(ROOT))}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"ARIMA": "complete", "rows": len(series["NVDA"]), "converged": evidence["converged"]}), flush=True)
        settings = protocol["lstm"]
        for variant, data in prepared.items():
            write_json(output / f"preprocessing_{variant}.json", data.provenance)
            for seed in settings["seeds"]:
                identity = f"{variant}_seed_{seed}"
                directory = output / identity
                directory.mkdir(exist_ok=False)
                record = {"identity": identity, "variant": variant, "seed": seed, "status": "started",
                          "started_at_utc": now(), "settings": settings,
                          "preprocessing": str((output / f"preprocessing_{variant}.json").relative_to(ROOT)),
                          "model": str((directory / "model.keras").relative_to(ROOT))}
                manifest["runs"].append(record)
                write_json(directory / "summary.json", record)
                write_json(output / "manifest.json", manifest)
                try:
                    model = build_model(settings, seed, lookback=30, features=len(VARIANTS[variant]))
                    training = train_model(model, data.X, data.y, settings, directory, validation=None)
                    require(training["epochs_completed"] == 19 and training["optimizer_steps"] == 1463
                            and training["trainable_parameters"] == settings["expected_parameters"][variant]
                            and training["best_epoch"] is None and not training["stopped_early"],
                            "Final fit violated fixed-epoch architecture invariants")
                    rows = [0, len(data.X) - 2, len(data.X) - 1]
                    inference = frozen_forecast(model, data.X[rows], data.origins[rows], data.correction_scaler)
                    record.update(training, status="complete", completed_at_utc=now(), check_indices=rows,
                                  check_target_dates=[str(data.metadata.target_date.iloc[i].date()) for i in rows],
                                  pretest_inference={key: value.tolist() for key, value in inference.items()})
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                    write_json(directory / "summary.json", record)
                    raise
                write_json(directory / "summary.json", record)
                write_json(output / "manifest.json", manifest)
                print(json.dumps({"identity": identity, "status": "complete", "epochs": 19,
                                  "seconds": training["elapsed_seconds"]}), flush=True)
        verify_hashes(protected)
        manifest.update(status="awaiting_verification", artifacts_sha256=artifact_hashes(output))
        write_json(output / "manifest.json", manifest)
        # A new interpreter reloads all checkpoints/scalers; no training is invoked there.
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "verify", "--record-verification"],
                       cwd=ROOT, check=True)
        verification = read_json(output / "verification.json")
        require(verification["status"] == "passed", "Independent artifact verification failed")
        manifest.update(status="frozen", frozen_at_utc=now(), artifacts_sha256=artifact_hashes(output),
                        verification="verification.json")
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": "frozen", "manifest": str(output / "manifest.json"),
                          "manifest_sha256": sha256(output / "manifest.json")}), flush=True)
    except Exception as error:
        if manifest.get("arima", {}).get("status") == "started":
            manifest["arima"].update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.update(status="failed", failed_at_utc=now(), error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc())
        write_json(output / "manifest.json", manifest)
        raise


def main():
    """Keep fitting, reload verification and sealed evaluation operationally separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "verify", "evaluate"))
    parser.add_argument("--record-verification", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate":
        raise SystemExit("2025 evaluation is sealed and requires separate explicit authorisation and implementation.")
    configure_runtime()
    if args.phase == "fit":
        require(not args.record_verification, "Verification recording belongs to the verification phase")
        fit_and_freeze()
    else:
        output = ROOT / "artifacts/final_training_v1"
        if args.record_verification:
            require(read_json(output / "manifest.json")["status"] == "awaiting_verification"
                    and not (output / "verification.json").exists(), "Frozen verification records cannot be overwritten")
        report = verify(output)
        if args.record_verification:
            write_json(output / "verification.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
