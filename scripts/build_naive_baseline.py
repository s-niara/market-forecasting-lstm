"""Build unscaled development windows and persist the naive validation benchmark."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from market_forecasting.dataset import build_development_dataset,naive_predictions,error_metrics

root=Path(__file__).resolve().parents[1]
config_path=root/'configs/nvda.json'; config=json.loads(config_path.read_text())
raw_path=root/'data/raw'/f"{config['target_ticker']}_yahoo_snapshot.json"
payload=json.loads(raw_path.read_text())['chart']['result'][0]
dates=pd.to_datetime(payload['timestamp'],unit='s',utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
# No test-window construction, scaling, tuning or performance evaluation.
close=pd.Series(payload['indicators']['quote'][0]['close'],index=dates)
data=build_development_dataset(close,config)
predictions=naive_predictions(data)

# Select by target partition, not by where the input window begins.
mask=data.metadata['split'].eq('validation').to_numpy()

# Persist a distinct run; keep generated arrays and vendor-derived predictions local.
output=root/'artifacts/naive_v1'
output.mkdir(exist_ok=False)
np.savez_compressed(output/'development_windows.npz',X=data.X,y=data.y)
data.metadata.to_json(output/'window_metadata.json',orient='records',date_format='iso')
table=data.metadata.loc[mask].copy()
table['actual_close']=data.y[mask];table['prediction']=predictions[mask]
table.to_json(output/'validation_predictions.json',orient='records',date_format='iso')

# Report errors in the unchanged price scale and identify the exact source inputs.
result={"model":"previous_session_close", "target":"Yahoo split-adjusted Close, retrospective USD basis",
        "input_shape":list(data.X.shape),"train_examples":int((~mask).sum()),"validation_examples":int(mask.sum()),
        "first_target":str(data.metadata.target_date.iloc[0].date()),
        "first_validation_origin":str(table.origin_date.iloc[0].date()),
        "first_validation_target":str(table.target_date.iloc[0].date()),
        "last_target":str(data.metadata.target_date.iloc[-1].date()),
        "validation_metrics":error_metrics(data.y[mask],predictions[mask]),
        "test_evaluated":False,"scaling":"none; naive forecast requires no fitted transform",
        "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "snapshot_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest()}
(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
