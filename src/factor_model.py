# src/factor_model.py
import numpy as np
import pandas as pd
from .config import RISK_FREE_RATE, TOP_N

def build_ff3_factors(dates) -> pd.DataFrame:
    """构造模拟 FF3 因子"""
    np.random.seed(42)
    ff3 = pd.DataFrame({
        "date": dates,
        "Mkt-RF": np.random.normal(0.008, 0.04, len(dates)),
        "SMB": np.random.normal(0.002, 0.02, len(dates)),
        "HML": np.random.normal(0.001, 0.02, len(dates)),
        "RF": [RISK_FREE_RATE] * len(dates)
    })
    ff3["date"] = pd.to_datetime(ff3["date"])
    return ff3

def ff3_regression(monthly: pd.DataFrame, ff3: pd.DataFrame) -> pd.DataFrame:
    """每只股票 OLS 回归"""
    demo_data = monthly.merge(ff3, on="date", how="left")
    demo_data["excess_ret"] = demo_data["monthly_ret"] - demo_data["RF"]
    
    results = []
    for symbol in demo_data["symbol"].unique():
        df = demo_data[demo_data["symbol"] == symbol].dropna()
        if len(df) < 10:
            continue
        X = np.column_stack([np.ones(len(df)), df["Mkt-RF"], df["SMB"], df["HML"]])
        Y = df["excess_ret"].values
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        results.append({
            "symbol": symbol, "alpha": beta[0], "beta_mkt": beta[1],
            "beta_smb": beta[2], "beta_hml": beta[3],
            "recent_ret": df["monthly_ret"].iloc[-1]
        })
    return pd.DataFrame(results)

def select_top_portfolio(factor_df: pd.DataFrame) -> pd.DataFrame:
    """选 Top N 组合"""
    factor_df["score"] = factor_df["alpha"] * 100 - factor_df["beta_mkt"] * 0.2
    top = factor_df.sort_values("score", ascending=False).head(TOP_N).copy()
    top["weight"] = top["score"] / top["score"].sum()
    return top