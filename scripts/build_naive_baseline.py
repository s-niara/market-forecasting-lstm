"""Build unscaled development windows and persist the naive validation benchmark."""
import hashlib
import json
from pathlib import Path

import numpy as np

from market_forecasting.calendar import audit_sessions
from market_forecasting.data import load_yahoo_snapshot, snapshot_provenance
from market_forecasting.dataset import build_development_dataset, naive_predictions, error_metrics


def main(root: Path) -> dict:
    """Validate development inputs and record a new, non-overwriting baseline run."""
    root = Path(root)
    output = root / "artifacts/naive_v1"
    # Refuse an existing run before reading data; original experiment evidence is immutable.
    if output.exists():
        raise FileExistsError(f"Baseline run already exists: {output}")
    config_path = root / "configs/nvda.json"
    config = json.loads(config_path.read_text())
    raw_path = root / "data/raw" / f"{config['target_ticker']}_yahoo_snapshot.json"
    provenance = snapshot_provenance(raw_path, root / "docs/data_verification.json", config["target_ticker"])

    # Select the permitted interval before decoding quotes so reserved targets remain unread.
    prices = load_yahoo_snapshot(raw_path, start=config["start"], end=config["validation_end"])
    audit = audit_sessions(prices.index, config["start"], config["validation_end"], "NASDAQ")
    # The generic window builder assumes adjacent rows are actual consecutive sessions.
    if audit["missing"] or audit["unexpected"] or audit["duplicate_dates"] or not audit["ordered"]:
        raise ValueError(f"Development session coverage is invalid: {audit}")
    data = build_development_dataset(prices["close"], config)
    predictions = naive_predictions(data)

    # Select by target partition, not by where the input window begins.
    mask = data.metadata["split"].eq("validation").to_numpy()
    table = data.metadata.loc[mask].copy()
    table["actual_close"] = data.y[mask]
    table["prediction"] = predictions[mask]
    result = {
        "model": "previous_session_close",
        "target": "Yahoo split-adjusted Close, retrospective USD basis",
        "input_shape": list(data.X.shape),
        "train_examples": int((~mask).sum()),
        "validation_examples": int(mask.sum()),
        "first_target": str(data.metadata.target_date.iloc[0].date()),
        "first_validation_origin": str(table.origin_date.iloc[0].date()),
        "first_validation_target": str(table.target_date.iloc[0].date()),
        "last_target": str(data.metadata.target_date.iloc[-1].date()),
        "validation_metrics": error_metrics(data.y[mask], predictions[mask]),
        "test_evaluated": False,
        "scaling": "none; naive forecast requires no fitted transform",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "snapshot_sha256": provenance["sha256"],
    }

    # All validation precedes writes; exclusive creation also guards against concurrent runs.
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / "development_windows.npz", X=data.X, y=data.y)
    data.metadata.to_json(output / "window_metadata.json", orient="records", date_format="iso")
    table.to_json(output / "validation_predictions.json", orient="records", date_format="iso")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
