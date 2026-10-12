"""Calendar coverage audit; no price-performance analysis."""
import pandas as pd
import pandas_market_calendars as mcal


def audit_sessions(index, start, end, calendar_name):
    dates = pd.DatetimeIndex(index)
    schedule = mcal.get_calendar(calendar_name).schedule(start_date=start, end_date=end)
    expected = schedule.index
    return {
        "calendar": calendar_name,
        "expected_sessions": len(expected),
        "observed_sessions": len(dates),
        "duplicate_dates": int(dates.duplicated().sum()),
        "ordered": bool(dates.is_monotonic_increasing),
        "missing": [str(d.date()) for d in expected.difference(dates)],
        "unexpected": [str(d.date()) for d in dates.difference(expected)],
        "sessions_by_year": {str(y): int(n) for y,n in schedule.groupby(schedule.index.year).size().items()},
    }
