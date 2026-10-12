"""Write a new calendar audit; existing audit files are never overwritten."""
import json
from pathlib import Path
from importlib.metadata import version
import pandas as pd
from market_forecasting.calendar import audit_sessions

root=Path(__file__).resolve().parents[1]
config=json.loads((root/'configs/nvda.json').read_text())
report={"calendar_library_version":version('pandas-market-calendars'),
        "scope":"Dates only, including test calendar; no performance inspection",
        "benchmark_calendar_note":"NYSE regular equity sessions used as proxy for NYSE Arca ETFs; not an official exchange-certified audit",
        "series":{}}
for ticker, calendar in [('NVDA','NASDAQ'),('SPY','NYSE'),('XLK','NYSE')]:
    payload=json.loads((root/'data/raw'/f'{ticker}_yahoo_snapshot.json').read_text())['chart']['result'][0]
    dates=pd.to_datetime(payload['timestamp'],unit='s',utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
    report['series'][ticker]=audit_sessions(dates,config['start'],str((pd.Timestamp(config['end_exclusive'])-pd.Timedelta(days=1)).date()),calendar)
path=root/'docs/calendar_audit_2026-10-12.json'
with path.open('x') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
