import json
from pathlib import Path
import pandas as pd
import pytest
from market_forecasting.data import validate_prices, load_yahoo_snapshot, load_training_panel


def payload():
    return {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1672689600, 1672776000],
        "indicators": {"quote": [{"open": [10., 11.], "high": [12., 13.],
          "low": [9., 10.], "close": [11., 12.], "volume": [100, 200]}],
          "adjclose": [{"adjclose": [10.5, 11.5]}]}}]}}


def write_sample(tmp_path):
    p=tmp_path/'sample.json'; p.write_text(json.dumps(payload())); return p


def test_exchange_date_and_separate_adjusted_close(tmp_path):
    frame=load_yahoo_snapshot(write_sample(tmp_path))
    assert frame.index[0] == pd.Timestamp('2023-01-02')
    assert frame['close'].iloc[0] == 11
    assert frame['adj_close'].iloc[0] == 10.5


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'range', 'volume'])
def test_invalid_records_rejected(tmp_path, fault):
    frame=load_yahoo_snapshot(write_sample(tmp_path))
    if fault=='duplicate': frame.index=[frame.index[0]]*2; frame.index=pd.DatetimeIndex(frame.index)
    if fault=='missing': frame.iloc[0, frame.columns.get_loc('close')]=float('nan')
    if fault=='range': frame.iloc[0, frame.columns.get_loc('high')]=1
    if fault=='volume': frame.iloc[0, frame.columns.get_loc('volume')]=-1
    with pytest.raises(ValueError): validate_prices(frame)


def test_training_panel_excludes_future_and_rejects_alignment(tmp_path):
    raw=tmp_path/'data/raw'; raw.mkdir(parents=True)
    for symbol in ['NVDA','SPY']:
        (raw/f'{symbol}_yahoo_snapshot.json').write_text(json.dumps(payload()))
    config={'target_ticker':'NVDA','context_tickers':['SPY'],'start':'2023-01-01','train_end':'2023-01-02'}
    panel=load_training_panel(tmp_path,config)
    assert all(len(f)==1 and f.index.max()==pd.Timestamp('2023-01-02') for f in panel.values())
    modified=payload(); modified['chart']['result'][0]['timestamp']=[1672776000,1672862400]
    (raw/'SPY_yahoo_snapshot.json').write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='Trading dates differ'):
        load_training_panel(tmp_path,config)
