import urllib.request, json, datetime, hashlib, math
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1]
out=root/'data/raw'
out.mkdir(parents=True,exist_ok=True)
if any(out.glob('*_yahoo_snapshot.json')):
 raise SystemExit('Frozen snapshots already exist. Use a new versioned data directory for a refresh.')
start=int(datetime.datetime(2015,1,1,tzinfo=datetime.timezone.utc).timestamp())
end=int(datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc).timestamp())
report={'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'series':{}}
datesets={}
for symbol in ['NVDA','SPY','XLK']:
 url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1d&events=div%2Csplits'
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=30) as r: raw=r.read()
 data=json.loads(raw)['chart']['result'][0]
 (out/f'{symbol}_yahoo_snapshot.json').write_bytes(raw)
 dates=[datetime.datetime.fromtimestamp(t,ZoneInfo('America/New_York')).date().isoformat() for t in data['timestamp']]
 q=data['indicators']['quote'][0]; adj=data['indicators']['adjclose'][0]['adjclose']
 fields={**q,'adjclose':adj}
 missing={k:sum(v is None or not math.isfinite(v) for v in vals) for k,vals in fields.items()}
 bad=[]
 for i,d in enumerate(dates):
  o,h,l,c,v=(q[k][i] for k in ['open','high','low','close','volume'])
  if None not in [o,h,l,c,v] and (min(o,h,l,c)<=0 or v<0 or l>min(o,c) or h<max(o,c) or l>h):bad.append(d)
 splits=[]
 for event in data.get('events',{}).get('splits',{}).values():
  day=datetime.datetime.fromtimestamp(event['date'],ZoneInfo('America/New_York')).date().isoformat()
  i=dates.index(day)
  splits.append({'date':day,'ratio':event['splitRatio'],'previous_close':q['close'][i-1],'close':q['close'][i],'close_return_pct':100*(q['close'][i]/q['close'][i-1]-1),'previous_volume':q['volume'][i-1],'volume':q['volume'][i]})
 report['series'][symbol]={'rows':len(dates),'first':dates[0],'last':dates[-1],'duplicate_dates':len(dates)-len(set(dates)),'ordered':dates==sorted(dates),'missing_nonfinite':missing,'invalid_ohlcv_dates':bad,'zero_volume_dates':[d for d,v in zip(dates,q['volume']) if v==0],'splits':splits,'dividend_events':len(data.get('events',{}).get('dividends',{})),'adjusted_close_ratio_range':[min(a/c for a,c in zip(adj,q['close'])),max(a/c for a,c in zip(adj,q['close']))],'sha256':hashlib.sha256(raw).hexdigest()}
 datesets[symbol]=set(dates)
report['date_alignment']={s:{'missing_vs_NVDA':sorted(datesets['NVDA']-d),'extra_vs_NVDA':sorted(d-datesets['NVDA'])} for s,d in datesets.items()}
(root/'docs/data_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
