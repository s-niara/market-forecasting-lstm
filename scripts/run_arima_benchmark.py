"""Run the bounded ARIMA benchmark on a verified snapshot; never overwrite a run."""
import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from market_forecasting.arima import run_arima_experiment
from market_forecasting.calendar import audit_sessions


def main():
    """Execute the ARIMA command-line benchmark and save a new local run."""
    # Forecast settings come from repository configuration; the CLI only names the run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="arima_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Run name must contain only letters, digits, underscores or hyphens")
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / args.run_name
    if output.exists():
        raise SystemExit(f"Run already exists: {output}; choose a new run name")
    config_path, settings_path = root / "configs/nvda.json", root / "configs/arima.json"
    config, settings = json.loads(config_path.read_text()), json.loads(settings_path.read_text())
    # Input provenance: fail on changed data instead of silently revising an experiment.
    snapshot = root / "data/raw" / f"{config['target_ticker']}_yahoo_snapshot.json"
    raw = snapshot.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    expected = json.loads((root / "docs/data_verification.json").read_text())
    if sha != expected["series"][config["target_ticker"]]["sha256"]:
        raise SystemExit("Snapshot checksum mismatch")
    payload = json.loads(raw)["chart"]["result"][0]
    dates = pd.to_datetime(payload["timestamp"], unit="s", utc=True).tz_convert(
        payload["meta"]["exchangeTimezoneName"]).normalize().tz_localize(None)
    # Truncate to development before any model fitting or price analysis.
    close = pd.Series(payload["indicators"]["quote"][0]["close"], index=dates).loc[
        config["start"]:config["validation_end"]]
    audit = audit_sessions(close.index, config["start"], config["validation_end"],
                           settings["calendar"])
    if audit["missing"] or audit["unexpected"] or audit["duplicate_dates"] or not audit["ordered"]:
        raise SystemExit("Development session coverage failed")
    # Fit on training data, then issue causal validation forecasts with fixed coefficients.
    experiment = run_arima_experiment(close, config, settings)
    summary = experiment.summary
    # Record configuration, numerical environment and exact source-file identities.
    summary.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config, arima_settings=settings, calendar_audit=audit,
        snapshot_sha256=sha,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        arima_config_sha256=hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        # Hash exact source bytes, including comments, to identify the recorded run.
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), root / "src/market_forecasting/arima.py",
                                 root / "src/market_forecasting/dataset.py",
                                 root / "src/market_forecasting/calendar.py"]},
        environment={"python": sys.version, "platform": platform.platform(),
                     **{name: version(name) for name in ["statsmodels", "numpy", "pandas", "scipy"]}})
    # Exclusive run creation keeps prior predictions and metadata intact.
    output.mkdir(parents=True, exist_ok=False)
    experiment.predictions.to_json(output / "validation_predictions.json", orient="records", date_format="iso")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["selected_order", "validation_metrics", "mae_improvement_pct"]}, indent=2))
    print(f"Saved run: {output}")


if __name__ == "__main__":
    main()
