"""Verify frozen snapshot identity without decoding historical prices."""
import json
from pathlib import Path
from market_forecasting.data import snapshot_provenance


def main():
    """Report exact checksum agreement for the configured source files."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/nvda.json").read_text())
    for symbol in [config["target_ticker"], *config["context_tickers"]]:
        path = root / "data/raw" / f"{symbol}_yahoo_snapshot.json"
        if not path.exists():
            raise SystemExit(f"Missing local snapshot: {path.name}; raw vendor data is not tracked in Git.")
        snapshot_provenance(path, root / "docs/data_verification.json", symbol)
        # Exact byte identity does not independently certify vendor adjustments.
        print(f"{symbol}: snapshot checksum verified (no quote values decoded)")


if __name__ == "__main__":
    main()
