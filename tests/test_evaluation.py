"""Synthetic verification of sealed evaluation, paired statistics and reusable plots."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest

from market_forecasting.evaluation import (SEEDS, VARIANTS, FORECAST_COLUMNS, validate_evaluation_table,
    evaluate_predictions, paired_circular_bootstrap, forecast_errors)
from market_forecasting import evaluation_plots as plots
from market_forecasting.features import calculate_features


def synthetic_table(n=40):
    dates = mcal.get_calendar('NASDAQ').schedule('2020-01-01','2020-05-31').index[:n+1]
    values = 100 + np.arange(n+1)*.12 + np.sin(np.arange(n+1)/2)
    frame = pd.DataFrame({'origin_date': dates[:-1], 'target_date': dates[1:],
                          'origin_close': values[:-1], 'actual_close': values[1:]})
    frame['naive'] = frame.origin_close
    frame['ARIMA'] = frame.origin_close + .05
    for v, shift in zip(VARIANTS, (.1,.2,-.1)):
        for seed, offset in zip(SEEDS, (-.1,0,.3)):
            frame[f'{v}_seed_{seed}'] = frame.origin_close + shift + offset
    return frame


def runner_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('evaluation_runner_test', root/'scripts/run_final_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_seed_first_not_prediction_ensemble():
    table = synthetic_table()
    for seed, error in zip(SEEDS, (-2.,0.,2.)):
        table[f'A_seed_{seed}'] = table.actual_close + error
    results = evaluate_predictions(table)
    assert results['families']['A']['mean_mae_usd'] == pytest.approx(4/3)
    assert results['families']['A']['mean_rmse_usd'] == pytest.approx(4/3)
    assert np.sqrt(np.mean((table[[f'A_seed_{s}' for s in SEEDS]].mean(axis=1)-table.actual_close)**2)) < 1e-12
    for comparison, left, right in [('B-A','B','A'),('C-B','C','B')]:
        expected=[]
        for seed in SEEDS:
            l = np.abs(table[f'{left}_seed_{seed}']-table.actual_close).mean()
            r = np.abs(table[f'{right}_seed_{seed}']-table.actual_close).mean()
            expected.append(l-r)
            assert results['paired_comparisons'][comparison]['per_seed'][str(seed)]['mae_difference_usd'] == pytest.approx(l-r)
        assert results['paired_comparisons'][comparison]['mean_mae_difference_usd'] == pytest.approx(np.mean(expected))
    naive = np.abs(table.naive-table.actual_close).mean()
    assert results['families']['A']['absolute_skill_usd']['mae'] == pytest.approx(naive-4/3)
    assert results['families']['A']['percentage_skill']['mae'] == pytest.approx(100*(1-(4/3)/naive))


def test_zero_baseline_skill_is_undefined():
    table = synthetic_table()
    table['origin_close'] = table['actual_close'] = table['naive'] = 100.
    result=evaluate_predictions(table)
    assert result['families']['A']['percentage_skill'] == {'mae':None,'rmse':None}
    assert result['per_model']['ARIMA']['percentage_skill'] == {'mae':None,'rmse':None}


@pytest.mark.parametrize('fault', ['missing_column','nonfinite','duplicate','gap','origin','naive','target_value','unsorted','timezone','empty'])
def test_invalid_comparisons_rejected(fault):
    table=synthetic_table()
    if fault=='missing_column':table=table.drop(columns='B_seed_123')
    if fault=='nonfinite':table.loc[3,'C_seed_42']=np.inf
    if fault=='duplicate':table.loc[2,'target_date']=table.loc[1,'target_date']
    if fault=='gap':table=table.drop(index=3)
    if fault=='origin':table.loc[0,'origin_date']=table.loc[0,'target_date']
    if fault=='naive':table.loc[0,'naive']+=1
    if fault=='target_value':table.loc[0,'actual_close']=-1
    if fault=='unsorted':table=table.iloc[::-1]
    if fault=='timezone':table['target_date']=table.target_date.dt.tz_localize('UTC')
    if fault=='empty':table=table.iloc[:0]
    with pytest.raises(ValueError):validate_evaluation_table(table)


def test_bootstrap_matches_independent_seed_mae_calculation():
    table=synthetic_table(25)
    actual=table.actual_close.to_numpy()
    result=paired_circular_bootstrap(table,(5,),repetitions=100,seed=314159)
    assert result==paired_circular_bootstrap(table,(5,),repetitions=100,seed=314159)
    rng=np.random.default_rng(314159); draws=[]
    for _ in range(100):
        starts=rng.integers(0,len(table),size=5)
        indices=np.concatenate([np.arange(start,start+5)%len(table) for start in starts])
        # Independently calculate each seed's MAE after common date resampling.
        b=np.mean([np.abs(table[f'B_seed_{s}'].to_numpy()[indices]-actual[indices]).mean() for s in SEEDS])
        a=np.mean([np.abs(table[f'A_seed_{s}'].to_numpy()[indices]-actual[indices]).mean() for s in SEEDS])
        draws.append(b-a)
    interval=result['intervals']['5']['B-A']
    assert interval['lower_usd']==pytest.approx(np.quantile(draws,.025))
    assert interval['upper_usd']==pytest.approx(np.quantile(draws,.975))
    assert result['joint_date_resampling_across_models_and_seeds']
    assert len(result['limitations'])==5


def test_joint_bootstrap_preserves_identical_model_contrasts():
    table=synthetic_table()
    for seed in SEEDS:table[f'B_seed_{seed}']=table[f'A_seed_{seed}']
    result=paired_circular_bootstrap(table,repetitions=100)
    for intervals in result['intervals'].values():
        assert intervals['B-A']=={'estimate_usd':0.,'lower_usd':0.,'upper_usd':0.}


@pytest.mark.parametrize('kwargs',[{'block_lengths':(50,)},{'block_lengths':(0,)},{'block_lengths':(5,5)},
                                   {'repetitions':1},{'seed':-1}])
def test_invalid_bootstrap_protocol_fails(kwargs):
    with pytest.raises(ValueError):paired_circular_bootstrap(synthetic_table(),**kwargs)


PLOT_FUNCTIONS = [plots.plot_actual_predictions,plots.plot_signed_errors,plots.plot_absolute_error_distributions,
                 plots.plot_naive_relative_performance,plots.plot_seed_variability,plots.plot_paired_differences,
                 plots.plot_cumulative_error_differences,plots.plot_forecast_examples]


@pytest.mark.parametrize('function',PLOT_FUNCTIONS)
def test_plots_use_actual_supplied_period_and_render(function):
    table=synthetic_table();table.index=np.arange(100,140)
    original=table.copy(deep=True)
    figure=function(table)
    figure.canvas.draw()
    assert len(figure.axes)>0
    pd.testing.assert_frame_equal(table,original)
    plt.close(figure)


def test_example_selection_is_shared_deterministic_and_not_model_dependent():
    table=synthetic_table()
    selected=plots.select_forecast_examples(table)
    ranking=table.assign(error=(table.naive-table.actual_close).abs()).sort_values(['error','target_date'],ascending=[False,True])
    expected=list(dict.fromkeys([table.target_date.iloc[0],*ranking.target_date.iloc[:3].tolist()]))
    assert selected==expected
    table['C_seed_42']+=100
    assert plots.select_forecast_examples(table)==selected


def test_explicit_history_boundary_retains_default_development_behavior():
    table=synthetic_table(40)
    dates=pd.DatetimeIndex([table.origin_date.iloc[0],*table.target_date])
    series={t:pd.Series(np.arange(len(dates))+100.,index=dates) for t in ['NVDA','SPY','XLK']}
    config={'start':'2020-01-01','validation_end':str(dates[25].date()),'test_end':str(dates[-1].date())}
    raw,features=calculate_features(series,config)
    explicit,explicit_features=calculate_features(series,config,history_end=dates[25])
    pd.testing.assert_frame_equal(raw,explicit);pd.testing.assert_frame_equal(features,explicit_features)
    changed={k:s.copy() for k,s in series.items()}
    for s in changed.values():s.loc[s.index>dates[25]]=9999
    _,later=calculate_features(changed,config,history_end=dates[25])
    pd.testing.assert_frame_equal(later,features)
    with pytest.raises(ValueError):calculate_features(series,config,history_end='2021-01-01')


def test_sealed_commands_refuse_before_runtime_or_model_access(monkeypatch):
    runner=runner_module(monkeypatch)
    monkeypatch.setattr(runner,'INDEPENDENT_EVALUATION_AUTHORISED',False)
    def forbidden(*args,**kwargs):raise AssertionError('Sealed command accessed the evaluation pipeline')
    monkeypatch.setattr(runner,'configure_runtime',forbidden)
    monkeypatch.setattr(runner,'frozen_evaluation_context',forbidden)
    monkeypatch.setattr(sys,'argv',['run_final_training.py','evaluate'])
    with pytest.raises(SystemExit,match='sealed'):runner.main()
    with pytest.raises(SystemExit,match='sealed'):runner.run_independent_evaluation({'execution_authorised':True})


def test_archived_source_identity_is_independent_of_later_live_code(monkeypatch,tmp_path):
    runner=runner_module(monkeypatch);monkeypatch.setattr(runner,'ROOT',tmp_path)
    original=b'original training source';path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('src/model.py',original)
    manifest={'source':{'source_archive':'source.zip','source_sha256':{'src/model.py':hashlib.sha256(original).hexdigest()}},
              'artifacts_sha256':{'source.zip':runner.sha256(path)}}
    (tmp_path/'src').mkdir();(tmp_path/'src/model.py').write_text('subsequent evaluation code')
    runner.verify_training_source_archive(manifest)
    manifest['source']['source_sha256']['src/model.py']='0'*64
    with pytest.raises(ValueError,match='source mismatch'):runner.verify_training_source_archive(manifest)


def synthetic_loop(monkeypatch, future_change=0., fail_record=False):
    runner=runner_module(monkeypatch)
    table=synthetic_table(4)
    history={t:pd.Series([float(table.origin_close.iloc[0])],index=pd.DatetimeIndex([table.origin_date.iloc[0]]))
             for t in ('NVDA','SPY','XLK')}
    class Model:
        def get_weights(self):return [np.array([1.])]
    class Cursor:
        parameters=np.array([.2])
        def observe(self,target,actual):events.append(('update',target))
    bundles={k:{'model':Model(),'preprocessing':{'fixed':1}} for k in FORECAST_COLUMNS if k not in ('naive','ARIMA')}
    events=[];issued=[]
    def forecast(history,config,origin,target,bundles,cursor,permitted_end):
        assert all(s.index[-1]==origin for s in history.values())
        close=float(history['NVDA'].iloc[-1])
        return {'origin_date':str(origin.date()),'target_date':str(target.date()),'origin_close':close,
                'predictions':{name:close for name in FORECAST_COLUMNS}}
    monkeypatch.setattr(runner,'issue_forecasts',forecast)
    def record(row):
        if fail_record:raise RuntimeError('recording failed')
        issued.append(row);events.append(('forecast',pd.Timestamp(row['target_date'])))
    def observe(date):
        assert events[-1]==('forecast',date)
        events.append(('observe',date))
        value=float(table.loc[table.target_date==date,'actual_close'].iloc[0])
        if date==table.target_date.iloc[-1]:value+=future_change
        return {ticker:value for ticker in history}
    def observed(row):events.append(('record_observation',pd.Timestamp(row['target_date'])))
    result=runner.rolling_evaluation(history,table.target_date,observe,record,observed,{},bundles,Cursor(),permitted_end='2020-12-31')
    return result,events,issued


def test_complete_loop_forecast_record_observe_update_order_and_future_invariance(monkeypatch):
    (table,_),events,issued=synthetic_loop(monkeypatch)
    assert [x[0] for x in events]==['forecast','observe','record_observation','update']*4
    (_, _),_,altered=synthetic_loop(monkeypatch,future_change=1000.)
    assert issued==altered
    assert table.target_date.is_monotonic_increasing and len(table)==4


def test_failed_forecast_record_prevents_observation(monkeypatch):
    with pytest.raises(RuntimeError,match='recording failed'):synthetic_loop(monkeypatch,fail_record=True)


def test_frozen_manifest_mismatch_stops_before_model_or_data_loading(monkeypatch,tmp_path):
    runner=runner_module(monkeypatch);monkeypatch.setattr(runner,'ROOT',tmp_path)
    directory=tmp_path/'artifacts/final_training_v1';directory.mkdir(parents=True)
    (directory/'manifest.json').write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('Integrity mismatch must stop before loading')
    monkeypatch.setattr(runner,'prepare_final',forbidden)
    with pytest.raises(ValueError,match='manifest integrity mismatch'):
        runner.frozen_evaluation_context({'frozen_manifest_sha256':'0'*64})


def test_synthetic_evaluation_records_forecasts_before_quote_decoding(monkeypatch,tmp_path):
    runner=runner_module(monkeypatch)
    table=synthetic_table(30)
    dates=pd.DatetimeIndex(table.target_date);origin=table.origin_date.iloc[0]
    history={ticker:pd.Series([float(table.origin_close.iloc[0])],index=pd.DatetimeIndex([origin]))
             for ticker in ('NVDA','SPY','XLK')}
    settings={'execution_authorised':True,'output_directory':'artifacts/synthetic_evaluation',
              'frozen_manifest_sha256':'synthetic',
              'protocol':{'test_start':str(dates[0].date()),'test_end':str(dates[-1].date()),'expected_calendar_sessions':30}}
    manifest={'protocol':{'data':{'calendars':{'NVDA':'NASDAQ','SPY':'NYSE','XLK':'NYSE'}}},'artifacts_sha256':{}}
    class Model:
        def get_weights(self):return [np.array([1.])]
    class Cursor:
        def __init__(self,*args,**kwargs):self.parameters=np.array([.2])
        def observe(self,target,actual):pass
    bundles={name:{'model':Model(),'preprocessing':{'fixed':1}} for name in FORECAST_COLUMNS if name not in ('naive','ARIMA')}
    monkeypatch.setattr(runner,'ROOT',tmp_path);(tmp_path/'artifacts').mkdir()
    monkeypatch.setattr(runner,'INDEPENDENT_EVALUATION_AUTHORISED',True)
    monkeypatch.setattr(runner,'frozen_evaluation_context',lambda settings:(manifest,{},history,bundles,None))
    monkeypatch.setattr(runner,'FrozenArima',Cursor)
    monkeypatch.setattr(runner,'load_yahoo_session_dates',lambda *args,**kwargs:dates)
    monkeypatch.setattr(runner,'snapshot_provenance',lambda *args,**kwargs:{})
    source=tmp_path/'synthetic.py';source.write_text('synthetic verification only')
    identity={'source_sha256':{'synthetic.py':runner.sha256(source)},'revision':'synthetic'}
    monkeypatch.setattr(runner,'evaluation_source_identity',lambda:identity)
    def forecast(history,config,origin,target,bundles,cursor,end):
        close=float(history['NVDA'].iloc[-1])
        return {'origin_date':str(origin.date()),'target_date':str(target.date()),'origin_close':close,
                'predictions':{name:close for name in FORECAST_COLUMNS}}
    monkeypatch.setattr(runner,'issue_forecasts',forecast)
    decoded=[]
    def quote_loader(path,*,start,end):
        saved=json.loads((tmp_path/settings['output_directory']/'evaluation.json').read_text())
        pending=saved['rows'][-1]
        assert pending['target_date']==str(start.date()) and start==end
        assert set(pending['predictions'])==set(FORECAST_COLUMNS)
        assert 'observed_close' not in pending
        decoded.append((path.name,start))
        value=float(table.loc[table.target_date==start,'actual_close'].iloc[0])
        return pd.DataFrame({'close':[value]},index=pd.DatetimeIndex([start]))
    monkeypatch.setattr(runner,'load_yahoo_snapshot',quote_loader)
    def record(table,observations,settings,manifest,*,scope):
        return {'status':'verified','scope':scope,'metrics':evaluate_predictions(table),
                'rows':len(table),'evaluation_implementation':identity}
    monkeypatch.setattr(runner,'evaluation_record',record)
    result=runner.run_independent_evaluation(settings)
    assert result['rows']==30 and len(decoded)==90
    assert all(date.year==2020 for _,date in decoded)
    assert result['metrics']['families']['A']['mean_mae_usd']==result['metrics']['per_model']['naive']['mae_usd']
    output=tmp_path/settings['output_directory']
    assert sorted(p.name for p in output.iterdir())==['evaluation.json','source.zip']
    with pytest.raises(ValueError,match='already exists'):runner.run_independent_evaluation(settings)


@pytest.mark.parametrize('code_gate,config_gate', [(False,False),(False,True),(True,False),(True,True)])
def test_independent_release_requires_both_controls(monkeypatch,code_gate,config_gate):
    runner=runner_module(monkeypatch)
    monkeypatch.setattr(runner,'INDEPENDENT_EVALUATION_AUTHORISED',code_gate)
    if code_gate and config_gate:
        runner.require_evaluation_authorisation({'execution_authorised':config_gate})
    else:
        with pytest.raises(SystemExit,match='sealed'):
            runner.require_evaluation_authorisation({'execution_authorised':config_gate})


def test_record_preserves_absolute_errors_without_changing_signed_errors(monkeypatch):
    runner=runner_module(monkeypatch)
    table=synthetic_table(30)
    settings=runner.evaluation_configuration()
    monkeypatch.setattr(runner,'evaluation_source_identity',lambda:{'revision':'synthetic'})
    manifest={'source':{'revision':'synthetic','source_archive':'source.zip'},
              'artifacts_sha256':{'source.zip':'synthetic'},'snapshot_sha256':{}}
    observations=[{'NVDA':float(x),'SPY':100.,'XLK':100.} for x in table.actual_close]
    record=runner.evaluation_record(table,observations,settings,manifest,scope='pretest_machinery_check')
    for i,row in enumerate(record['rows']):
        for name in FORECAST_COLUMNS:
            expected=float(table[name].iloc[i]-table.actual_close.iloc[i])
            assert row['errors'][name]==expected
            assert row['absolute_errors'][name]==abs(expected)
