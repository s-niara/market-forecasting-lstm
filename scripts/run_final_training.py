"""Fit and freeze pre-test models, then evaluate only through explicit release controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import copy
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
from market_forecasting.data import load_yahoo_snapshot, load_yahoo_session_dates, snapshot_provenance
from market_forecasting.evaluation import (FORECAST_COLUMNS, validate_evaluation_table,
    evaluate_predictions, paired_circular_bootstrap, forecast_errors)
from market_forecasting.features import (VARIANTS, calculate_features, prepare_final_feature_dataset,
                                         prepare_origin_input, validate_feature_preprocessing)
from market_forecasting.lstm import (configure_runtime, build_model, train_model,
                                     load_frozen_model, frozen_forecast)
from market_forecasting.preprocessing import Standardisation
from run_lstm_experiments import load_stage, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/final_training_v1.json"
SEALED_BOUNDARY = pd.Timestamp("2024-12-31")
FROZEN_MANIFEST_SHA256 = "e2a46cc211c393b197f67f275ea214dabc7fd67a74dae90401be9e5ec2bd88dc"
EVALUATION_CONFIG_PATH = ROOT / "configs/final_evaluation_v1.json"
# Milestone 8A explicitly authorises independent evaluation of the frozen methodology.
INDEPENDENT_EVALUATION_AUTHORISED = True


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
    verify_training_source_archive(manifest)
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



def verify_training_source_archive(manifest):
    """Verify the frozen training implementation independently of later evaluation code."""
    expected = manifest["source"]["source_sha256"]
    path = ROOT / manifest["source"]["source_archive"]
    require(sha256(path) == manifest["artifacts_sha256"][str(path.relative_to(ROOT))],
            "Frozen training source archive integrity mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected),
                "Training source archive membership mismatch")
        for name in names:
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "Invalid training source archive path")
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected[name],
                    f"Frozen training source mismatch: {name}")


def evaluation_configuration():
    """Bind evaluation settings to the approved training protocol without changing it."""
    settings = read_json(EVALUATION_CONFIG_PATH)
    protocol, _ = approved_configuration()
    require(settings["frozen_manifest_sha256"] == FROZEN_MANIFEST_SHA256,
            "Unexpected frozen training identity")
    scientific_protocol = {key: value for key, value in protocol["evaluation_protocol"].items()
                           if key not in {"test_sealed", "test_access_authorised", "execution_authorised"}}
    require(settings["protocol"] == scientific_protocol
            and settings["comparison"] == protocol["comparison"], "Frozen evaluation protocol differs")
    require(settings["seeds"] == [42, 123, 2026] and settings["output_directory"] == "artifacts/final_evaluation_v1"
            and settings["dry_run_sessions"] == 30, "Evaluation identities or destinations differ")
    return settings


def require_evaluation_authorisation(settings):
    """Keep real test access closed independently of the configuration file."""
    if not INDEPENDENT_EVALUATION_AUTHORISED or settings.get("execution_authorised") is not True:
        raise SystemExit("2025 evaluation remains sealed; explicit release authorisation is required.")


def frozen_evaluation_context(settings):
    """Verify frozen payloads and load models with their original preprocessing."""
    output = ROOT / "artifacts/final_training_v1"
    require(sha256(output / "manifest.json") == settings["frozen_manifest_sha256"],
            "Frozen training manifest integrity mismatch")
    manifest = read_json(output / "manifest.json")
    require(manifest["status"] == "frozen", "Final models have not been frozen")
    verify_hashes(manifest["artifacts_sha256"])
    verify_hashes(manifest["protected_sha256"])
    verify_training_source_archive(manifest)
    protocol, config = approved_configuration()
    series, hashes, prepared = prepare_final(protocol, config)
    require(hashes == manifest["snapshot_sha256"], "Frozen source identity mismatch")
    bundles = {}
    for record in manifest["runs"]:
        identity, variant, seed = record["identity"], record["variant"], record["seed"]
        require(identity == f"{variant}_seed_{seed}", "Frozen run identity mismatch")
        preprocessing = read_json(output / f"preprocessing_{variant}.json")
        require(preprocessing == prepared[variant].provenance, "Saved preprocessing reconstruction mismatch")
        model = load_frozen_model(output / identity / "model.keras", VARIANTS[variant], protocol["lstm"])
        require(model.layers[1].get_config()["seed"] == seed, "Frozen seed metadata mismatch")
        bundles[identity] = {"variant": variant, "model": model, "preprocessing": preprocessing,
                             "correction_scaler": Standardisation(**preprocessing["correction_scaler"])}
    require(set(bundles) == set(FORECAST_COLUMNS) - {"naive", "ARIMA"}, "Incomplete frozen model set")
    fitted = ARIMAResults.load(output / "arima.pkl")
    require(arima_checks(fitted) == manifest["arima"]["checks"], "Frozen ARIMA state mismatch")
    return manifest, config, series, bundles, fitted


def issue_forecasts(history, config, origin, target, bundles, cursor, permitted_end):
    """Forecast from the observed prefix, reusing each variant's transformed window."""
    origin, target, end = map(pd.Timestamp, (origin, target, permitted_end))
    require(origin < target <= end, "Forecast dates exceed the declared evaluation boundary")
    require(set(bundles) == set(FORECAST_COLUMNS) - {"naive", "ARIMA"}, "Every approved seed is required")
    audit = audit_sessions(pd.DatetimeIndex([origin, target]), origin, target, "NASDAQ")
    require(not audit["missing"] and not audit["unexpected"] and audit["expected_sessions"] == 2,
            "Forecast horizon must be exactly one exchange session")
    require(all(s.index[-1] == origin for s in history.values()), "History must stop at the forecast origin")
    transformed = {}
    predictions = {}
    for identity, bundle in bundles.items():
        variant = bundle["variant"]
        if variant not in transformed:
            transformed[variant] = prepare_origin_input(history, config, variant, origin,
                bundle["preprocessing"], permitted_end=end)
        X, close, _ = transformed[variant]
        # Even shared transforms remain tied to each checkpoint's ordered metadata.
        validate_feature_preprocessing(bundle["preprocessing"], variant)
        require(bundle["preprocessing"] == next(b["preprocessing"] for b in bundles.values()
                if b["variant"] == variant), "Seed preprocessing differs within a feature family")
        result = frozen_forecast(bundle["model"], X, np.array([close]), bundle["correction_scaler"])
        predictions[identity] = float(result["prediction"][0])
        predictions.setdefault("naive", close)
        require(predictions["naive"] == close, "Model forecast origins differ")
    predictions["ARIMA"] = cursor.forecast(origin, target)
    require(np.isfinite(list(predictions.values())).all(), "Nonfinite issued forecast")
    return {"origin_date": str(origin.date()), "target_date": str(target.date()),
            "origin_close": predictions["naive"], "predictions": predictions}


def rolling_evaluation(history, target_dates, observation_provider, record_forecast, record_observation,
                       config, bundles, cursor, *, permitted_end):
    """Record all forecasts before revealing and incorporating each target observation."""
    dates = pd.DatetimeIndex(target_dates)
    require(len(dates) > 0 and not dates.hasnans and not dates.has_duplicates
            and dates.is_monotonic_increasing and dates.tz is None
            and dates.equals(dates.normalize()) and dates[-1] <= pd.Timestamp(permitted_end),
            "Invalid evaluation target sequence")
    history = {ticker: s.copy(deep=True) for ticker, s in history.items()}
    require(set(history) == {"NVDA", "SPY", "XLK"}, "All approved instruments are required")
    require(all(isinstance(s, pd.Series) and len(s) > 0 for s in history.values()),
            "Nonempty historical Close series are required")
    origin = history["NVDA"].index[-1]
    for ticker, calendar in (("NVDA", "NASDAQ"), ("SPY", "NYSE"), ("XLK", "NYSE")):
        require(history[ticker].index.equals(history["NVDA"].index), "Historical instrument dates differ")
        coverage = audit_sessions(pd.DatetimeIndex([origin]).append(dates), origin, dates[-1], calendar)
        require(not coverage["missing"] and not coverage["unexpected"], "Missing or inconsistent evaluation sessions")
    before = {key: json.dumps(bundle["preprocessing"], sort_keys=True) for key, bundle in bundles.items()}
    weights = {key: [w.copy() for w in bundle["model"].get_weights()] for key, bundle in bundles.items()}
    coefficients = cursor.parameters.copy()
    rows, observations = [], []
    for target in dates:
        issued = issue_forecasts(history, config, origin, target, bundles, cursor, permitted_end)
        # Durable record callbacks can fail closed before the observation provider is called.
        record_forecast(copy.deepcopy(issued))
        observed = observation_provider(target)
        require(set(observed) == {"NVDA", "SPY", "XLK"}, "Observation must contain all aligned instruments")
        require(all(np.isscalar(value) and not isinstance(value, (str, bool))
                    and np.isfinite(value) and value > 0 for value in observed.values()),
                "Observed prices must be finite positive scalars")
        row = {key: issued[key] for key in ("origin_date", "target_date", "origin_close")}
        row.update(actual_close=float(observed["NVDA"]), **issued["predictions"])
        errors = {name: row[name] - row["actual_close"] for name in FORECAST_COLUMNS}
        record_observation({"target_date": str(target.date()), "observed_close": dict(observed), "errors": errors})
        cursor.observe(target, observed["NVDA"])
        for ticker in history:
            history[ticker].loc[target] = float(observed[ticker])
        rows.append(row)
        observations.append(dict(observed))
        origin = target
    for key, bundle in bundles.items():
        require(json.dumps(bundle["preprocessing"], sort_keys=True) == before[key], "Input scaler metadata changed")
        require(all(np.array_equal(a, b) for a, b in zip(weights[key], bundle["model"].get_weights())),
                "Frozen model weights changed")
    np.testing.assert_array_equal(cursor.parameters, coefficients)
    table = validate_evaluation_table(pd.DataFrame(rows))
    return table, observations


def evaluation_source_identity():
    """Identify the evaluation implementation separately from the archived training code."""
    files = []
    for directory in ("src", "scripts", "tests", "configs"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json"))
    files.extend(p for p in ROOT.glob("*") if p.is_file() and
                 (p.name.startswith("requirements") or p.name == "pyproject.toml"))
    return {"revision": subprocess.check_output(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_status": subprocess.check_output(["git", "--no-optional-locks", "status", "--short"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files}}


def evaluation_record(table, observations, settings, manifest, *, scope):
    """Build one compact record of forecasts, errors, statistics and exact provenance."""
    results = evaluate_predictions(table)
    b = settings["protocol"]["bootstrap"]
    bootstrap = paired_circular_bootstrap(table, (b["primary_block_length"], *b["sensitivity_block_lengths"]),
                                         b["resamples"], b["seed"])
    errors = forecast_errors(table)
    rows = []
    for i, row in table.iterrows():
        rows.append({"origin_date": str(row.origin_date.date()), "target_date": str(row.target_date.date()),
                     "origin_close": float(row.origin_close), "observed_close": observations[i],
                     "predictions": {name: float(row[name]) for name in FORECAST_COLUMNS},
                     "errors": {name: float(errors[name].iloc[i]) for name in FORECAST_COLUMNS},
                     "absolute_errors": {name: float(abs(errors[name].iloc[i])) for name in FORECAST_COLUMNS}})
    from market_forecasting.evaluation_plots import select_forecast_examples
    return {"version": 1, "status": "verified", "scope": scope, "recorded_at_utc": now(),
            "configuration": settings, "configuration_sha256": sha256(EVALUATION_CONFIG_PATH),
            "training_manifest_sha256": settings["frozen_manifest_sha256"],
            "training_source_revision": manifest["source"]["revision"],
            "training_source_archive_sha256": manifest["artifacts_sha256"][manifest["source"]["source_archive"]],
            "frozen_payload_sha256": manifest["artifacts_sha256"],
            "evaluation_implementation": evaluation_source_identity(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                **{name: version(name) for name in ("tensorflow", "keras", "numpy", "pandas", "statsmodels",
                                                    "matplotlib", "pandas_market_calendars")}},
            "snapshot_sha256": manifest["snapshot_sha256"], "rows": rows,
            "metrics": results, "bootstrap": bootstrap,
            "representative_seed": 42,
            "illustrative_dates": [str(date.date()) for date in select_forecast_examples(table)],
            "verification": {"forecast_recorded_before_observation": True, "shared_target_dates": True,
                             "frozen_models_and_preprocessing": True, "artifact_checksums_match": True},
            "interpretation": ("Pre-test machinery demonstration using parameters fitted through the demonstrated period; not holdout accuracy."
                               if scope == "pretest_machinery_check" else "Independent evaluation of prespecified frozen models.")}


def pretest_dry_run():
    """Exercise the complete evaluation machinery on thirty permitted pre-test sessions."""
    settings = evaluation_configuration()
    manifest, config, series, bundles, fitted = frozen_evaluation_context(settings)
    dates = series["NVDA"].index[-settings["dry_run_sessions"]:]
    origin = series["NVDA"].index[-settings["dry_run_sessions"] - 1]
    history = {ticker: s.loc[:origin].copy() for ticker, s in series.items()}
    # Refilter the earlier prefix without refitting coefficients; this checks mechanics only.
    prefix = fitted.apply(history["NVDA"].to_numpy(), refit=False)
    cursor = FrozenArima(prefix, origin)
    trace = []
    def record_forecast(issued):
        trace.append((issued["target_date"], "forecast"))
    def observe(target):
        require(trace[-1] == (str(target.date()), "forecast"), "Observation preceded forecast recording")
        return {ticker: float(series[ticker].loc[target]) for ticker in series}
    def record_observation(observed):
        trace.append((observed["target_date"], "observation"))
    table, observations = rolling_evaluation(history, dates, observe, record_forecast, record_observation,
                                             config, bundles, cursor, permitted_end="2024-12-31")
    require(len(trace) == 2 * len(dates), "Incomplete forecast/observation trace")
    verify_hashes(manifest["artifacts_sha256"])
    record = evaluation_record(table, observations, settings, manifest, scope="pretest_machinery_check")
    record["verification"]["event_count"] = len(trace)
    return record


def run_independent_evaluation(settings):
    """Run the complete independent evaluation only after the separate release gate."""
    require_evaluation_authorisation(settings)
    manifest, config, history, bundles, fitted = frozen_evaluation_context(settings)
    period = settings["protocol"]
    dates = None
    for ticker, calendar in manifest["protocol"]["data"]["calendars"].items():
        snapshot = ROOT / "data/raw" / f"{ticker}_yahoo_snapshot.json"
        actual_dates = load_yahoo_session_dates(snapshot, start=period["test_start"], end=period["test_end"])
        coverage = audit_sessions(actual_dates, period["test_start"], period["test_end"], calendar)
        require(not coverage["missing"] and not coverage["unexpected"] and not coverage["duplicate_dates"]
                and coverage["ordered"] and len(actual_dates) == period["expected_calendar_sessions"],
                "Independent period has missing or inconsistent sessions")
        require(dates is None or dates.equals(actual_dates), "Independent instrument dates differ")
        dates = actual_dates
    output = ROOT / settings["output_directory"]
    require(not output.exists(), "Independent evaluation evidence already exists; no overwrite is allowed")
    output.mkdir(exist_ok=False)
    path = output / "evaluation.json"
    implementation = evaluation_source_identity()
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in implementation["source_sha256"]:
            archive.write(ROOT / relative, relative)
    source_digest = sha256(output / "source.zip")
    started_at = now()
    record = {"status": "started", "scope": "independent_test", "started_at_utc": started_at,
              "configuration": settings, "training_manifest_sha256": settings["frozen_manifest_sha256"],
              "evaluation_implementation": implementation, "evaluation_source_archive_sha256": source_digest, "rows": []}
    write_json(path, record)
    cursor = FrozenArima(fitted, history["NVDA"].index[-1], permitted_end=period["test_end"])
    def record_forecast(issued):
        record["rows"].append(issued)
        write_json(path, record)
    def observe(target):
        # Quotes are decoded only for the session whose forecasts are already on disk.
        observed = {}
        for ticker in ("NVDA", "SPY", "XLK"):
            snapshot = ROOT / "data/raw" / f"{ticker}_yahoo_snapshot.json"
            frame = load_yahoo_snapshot(snapshot, start=target, end=target)
            require(len(frame) == 1 and frame.index[0] == target, "Observation date mismatch")
            observed[ticker] = float(frame.close.iloc[0])
        return observed
    def record_observation(observed):
        require(record["rows"][-1]["target_date"] == observed["target_date"], "Observation alignment failed")
        record["rows"][-1].update(observed_close=observed["observed_close"], errors=observed["errors"])
        write_json(path, record)
    try:
        table, observations = rolling_evaluation(history, dates, observe, record_forecast, record_observation,
            config, bundles, cursor, permitted_end=period["test_end"])
        verify_hashes(manifest["artifacts_sha256"])
        for ticker in ("NVDA", "SPY", "XLK"):
            snapshot_provenance(ROOT / "data/raw" / f"{ticker}_yahoo_snapshot.json",
                                ROOT / "docs/data_verification.json", ticker)
        record = evaluation_record(table, observations, settings, manifest, scope="independent_test")
        require(record["evaluation_implementation"]["source_sha256"] == implementation["source_sha256"],
                "Evaluation implementation changed during execution")
        record["evaluation_implementation"] = implementation
        record["evaluation_source_archive_sha256"] = source_digest
        record["execution"] = {"started_at_utc": started_at, "completed_at_utc": now(),
                               "target_count": len(table), "forecast_count": len(table) * len(FORECAST_COLUMNS)}
        write_json(path, record)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(path, record)
        raise
    return record


def main():
    """Separate fitting, frozen verification, pre-test demonstration and sealed testing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "verify", "dry-run", "evaluate"))
    parser.add_argument("--record-verification", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate":
        if not INDEPENDENT_EVALUATION_AUTHORISED:
            raise SystemExit("2025 evaluation remains sealed; explicit release authorisation is required.")
        settings = evaluation_configuration()
        require_evaluation_authorisation(settings)
        configure_runtime()
        report = run_independent_evaluation(settings)
    elif args.phase == "dry-run":
        require(not args.record_verification, "Dry-run does not modify frozen verification artifacts")
        configure_runtime()
        report = pretest_dry_run()
    elif args.phase == "fit":
        require(not args.record_verification, "Verification recording belongs to the verification phase")
        configure_runtime()
        fit_and_freeze()
        return
    else:
        configure_runtime()
        output = ROOT / "artifacts/final_training_v1"
        if args.record_verification:
            require(read_json(output / "manifest.json")["status"] == "awaiting_verification"
                    and not (output / "verification.json").exists(), "Frozen verification records cannot be overwritten")
        report = verify(output)
        if args.record_verification:
            write_json(output / "verification.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
