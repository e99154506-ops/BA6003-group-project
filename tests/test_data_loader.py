import pytest
import pandas as pd
from src.data_loader import compute_monthly_returns

def test_compute_monthly_returns():
    # 构造假数据
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=60, freq="D"),
        "symbol": ["AAPL"] * 30 + ["MSFT"] * 30,
        "close": list(range(100, 130)) + list(range(200, 230))
    })
    result = compute_monthly_returns(df)
    assert "monthly_ret" in result.columns
    assert len(result) > 0
    assert result["monthly_ret"].notna().any()