import pytest
import pandas as pd
from src.backtest_model import run_backtest

def test_run_backtest_basic():
    monthly = pd.DataFrame({"date": pd.to_datetime(["2023-01-31"]*2), "symbol": ["A","B"], "monthly_ret": [0.01, 0.02]})
    top = pd.DataFrame({"symbol": ["A","B"], "weight": [0.5, 0.5]})
    result = run_backtest(monthly, top) # 假设返回累计收益DF
    assert not result.empty