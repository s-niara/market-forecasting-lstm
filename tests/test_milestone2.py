"""Synthetic checks for window alignment, date boundaries and benchmark metrics."""
import numpy as np
import pandas as pd
import pytest
from market_forecasting.calendar import audit_sessions
from market_forecasting.dataset import build_development_dataset,naive_predictions,error_metrics


def config():
    """Return short synthetic split boundaries and a two-session lookback."""
    return dict(price_convention='split_adjusted_close',start='2023-01-01',train_end='2023-01-05',validation_end='2023-01-10',test_end='2023-01-20',forecast_horizon_trading_days=1,initial_sequence_length=2)


def test_window_alignment_and_boundary_context():
    """Verify input/target offsets and legitimate context across a split boundary."""
    dates=pd.bdate_range('2023-01-02','2023-01-13')
    values=pd.Series(np.arange(1,len(dates)+1,dtype=float),index=dates)
    ds=build_development_dataset(values,config())
    np.testing.assert_array_equal(ds.X[0,:,0],[1,2])
    assert ds.y[0]==3
    assert (ds.metadata.origin_date < ds.metadata.target_date).all()
    first=ds.metadata[ds.metadata.split=='validation'].iloc[0]
    assert first.origin_date==pd.Timestamp('2023-01-05')
    assert first.target_date==pd.Timestamp('2023-01-06')
    np.testing.assert_array_equal(naive_predictions(ds),ds.X[:,-1,0])


def test_test_values_cannot_change_development_windows():
    """Check that post-validation price changes cannot affect development arrays."""
    dates=pd.bdate_range('2023-01-02','2023-01-13')
    values=pd.Series(np.arange(1,len(dates)+1,dtype=float),index=dates)
    a=build_development_dataset(values,config())
    values.loc['2023-01-11':]=np.nan
    b=build_development_dataset(values,config())
    np.testing.assert_array_equal(a.X,b.X);np.testing.assert_array_equal(a.y,b.y)
    assert b.metadata.target_date.max()<=pd.Timestamp(config()['validation_end'])


def test_missing_session_and_unexpected_day():
    """Detect a missing session and an incorrectly included market holiday."""
    audit=audit_sessions(pd.DatetimeIndex(['2024-07-03','2024-07-04']),'2024-07-03','2024-07-05','NASDAQ')
    assert audit['missing']==['2024-07-05']
    assert audit['unexpected']==['2024-07-04']


def test_special_closure():
    """Verify the calendar excludes the January 2025 special market closure."""
    audit=audit_sessions(pd.DatetimeIndex(['2025-01-08','2025-01-10']),'2025-01-08','2025-01-10','NASDAQ')
    assert audit['missing']==[] and audit['unexpected']==[]


def test_metrics_known_values():
    """Compare errors with hand-computed values and reject nonfinite predictions."""
    m=error_metrics([1,3],[2,1]);assert m['mae_usd']==1.5
    assert m['rmse_usd']==pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError):error_metrics([1],[float('nan')])


def test_unapproved_convention_rejected():
    """Reject a target convention that is not the implemented split-adjusted Close."""
    c=config();c['price_convention']='unresolved'
    with pytest.raises(ValueError,match='explicitly agreed'):
        build_development_dataset(pd.Series([1.,2.,3.],index=pd.bdate_range('2023-01-02',periods=3)),c)
