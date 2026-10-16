"""Display a training-period session audit without decoding quote values."""
import json
from importlib.metadata import version
from pathlib import Path
from market_forecasting.calendar import audit_sessions
from market_forecasting.data import load_yahoo_session_dates


def main():
    """Compare configured training dates with exchange-calendar sessions."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/nvda.json").read_text())
    report = {"calendar_library_version": version("pandas-market-calendars"),
              "scope": {"start": config["start"], "end": config["train_end"], "quote_values_decoded": False},
              "benchmark_calendar_note": "NYSE regular equity sessions are a proxy for NYSE Arca ETFs, not exchange certification",
              "series": {}}
    for symbol, calendar in [("NVDA", "NASDAQ"), ("SPY", "NYSE"), ("XLK", "NYSE")]:
        dates = load_yahoo_session_dates(root / "data/raw" / f"{symbol}_yahoo_snapshot.json",
                                        start=config["start"], end=config["train_end"])
        report["series"][symbol] = audit_sessions(dates, config["start"], config["train_end"], calendar)
    # Display the audit without replacing the preserved historical report.
    print(json.dumps(report, indent=2))
    if any(a["missing"] or a["unexpected"] or a["duplicate_dates"] or not a["ordered"]
           for a in report["series"].values()):
        raise SystemExit("Training calendar audit failed; investigate without filling records")


if __name__ == "__main__":
    main()
