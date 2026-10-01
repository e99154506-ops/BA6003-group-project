import pytest
import pandas as pd
import numpy as np
from src.factor_model import build_ff3_factors, ff3_regression, select_top_portfolio

def test_build_ff3_factors():
    dates = pd.date_range("2023-01-01", periods=12, freq="ME")
    ff3 = build_ff3_factors(dates)
    assert len(ff3) == 12
    assert "Mkt-RF" in ff3.columns

def test_select_top_portfolio():
    factor_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "alpha": [0.01, 0.02, 0.005, 0.015],
        "beta_mkt": [1.0, 0.8, 1.2, 0.9]
    })
    top = select_top_portfolio(factor_df)
    assert len(top) == 3
    assert "weight" in top.columns
    assert abs(top["weight"].sum() - 1.0) < 1e-6