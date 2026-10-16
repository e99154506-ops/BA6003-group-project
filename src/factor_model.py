import numpy as np
import pandas as pd
from .config import RISK_FREE_RATE, TOP_N, TOP_QUANTILE, LOOKBACK

# ==================== 价格因子 ====================

def build_price_factors(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """
    从月末收盘价计算动量因子
    返回长表: date, symbol, momentum_6m
    """
    # 6个月动量 = 当前价格 / 6个月前价格 - 1
    momentum = monthly_close.pct_change(LOOKBACK)
    
    mom_long = momentum.stack().reset_index()
    mom_long.columns = ["date", "symbol", "momentum_6m"]
    
    # 6个月波动率（用月收益滚动标准差）
    monthly_ret = monthly_close.pct_change()
    volatility = monthly_ret.rolling(LOOKBACK).std() * np.sqrt(LOOKBACK)
    vol_long = volatility.stack().reset_index()
    vol_long.columns = ["date", "symbol", "volatility_6m"]
    
    # 合并
    factors = mom_long.merge(vol_long, on=["date", "symbol"], how="left")
    return factors.dropna(subset=["momentum_6m"])


# ==================== FF3 模拟因子 ====================

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


# ==================== FF3 回归 ====================

def ff3_regression(monthly_ret_wide: pd.DataFrame, ff3: pd.DataFrame) -> pd.DataFrame:
    """
    每只股票对 FF3 做 OLS，返回 α 和 β
    """
    monthly = monthly_ret_wide.stack().reset_index()
    monthly.columns = ["date", "symbol", "monthly_ret"]
    monthly["date"] = pd.to_datetime(monthly["date"])
    
    demo_data = monthly.merge(ff3, on="date", how="left")
    demo_data["excess_ret"] = demo_data["monthly_ret"] - demo_data["RF"]
    
    results = []
    for symbol in demo_data["symbol"].unique():
        df = demo_data[demo_data["symbol"] == symbol].dropna()
        if len(df) < 10:
            continue
        X = np.column_stack([np.ones(len(df)), df["Mkt-RF"], df["SMB"], df["HML"]])
        Y = df["excess_ret"].values
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            results.append({
                "symbol": symbol,
                "alpha": beta[0] * 12,   # 年化 α
                "beta_mkt": beta[1],
                "beta_smb": beta[2],
                "beta_hml": beta[3],
            })
        except Exception:
            continue
    
    return pd.DataFrame(results)


# ==================== 选股 ====================

def select_top_portfolio(factor_df: pd.DataFrame, method: str = "combo") -> pd.DataFrame:
    """
    选股方法:
    - "alpha": FF3 α 打分
    - "momentum": 纯动量
    - "combo": 混合打分
    """
    if method == "alpha":
        factor_df["score"] = factor_df["alpha"] * 100 - factor_df["beta_mkt"] * 0.2
        top_n = TOP_N if TOP_N > 1 else max(1, int(len(factor_df) * TOP_QUANTILE))
        top = factor_df.sort_values("score", ascending=False).head(top_n).copy()
    
    elif method == "momentum":
        top_n = max(1, int(len(factor_df) * TOP_QUANTILE))
        top = factor_df.sort_values("momentum_6m", ascending=False).head(top_n).copy()
        top["score"] = top["momentum_6m"]
    
    elif method == "combo":
        factor_df["score"] = (
            factor_df.get("alpha", 0) * 50 +
            factor_df.get("momentum_6m", 0) * 0.5 -
            factor_df.get("volatility_6m", 0) * 0.3
        )
        top_n = max(1, int(len(factor_df) * TOP_QUANTILE))
        top = factor_df.sort_values("score", ascending=False).head(top_n).copy()
    
    else:
        raise ValueError(f"未知 method: {method}")
    
    top["weight"] = top["score"] / top["score"].sum()
    return top