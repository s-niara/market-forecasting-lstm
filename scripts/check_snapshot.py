"""Verify frozen snapshot identity before reproducing an experiment."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]

# Identity checks do not independently validate the vendor adjustment methodology.
audit = json.loads((root / "docs/data_verification.json").read_text())
for ticker, info in audit["series"].items():
    path = root / "data/raw" / f"{ticker}_yahoo_snapshot.json"
    if not path.exists():
        raise SystemExit(f"Missing local snapshot: {path.name}; raw vendor data is not tracked in Git.")
    # Hash raw bytes, not a reserialised object, to detect any snapshot change.
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != info["sha256"]:
        raise SystemExit(f"Checksum mismatch: {ticker}")
    result = json.loads(path.read_text())["chart"]["result"][0]
    if len(result["timestamp"]) != info["rows"]:
        raise SystemExit(f"Row count mismatch: {ticker}")
    print(f"{ticker}: checksum and {info['rows']} rows verified")
