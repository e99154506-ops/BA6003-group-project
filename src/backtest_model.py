# src/backtest.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import OUTPUTS_DIR
import logging
logger = logging.getLogger(__name__)
# 在函数内加：logger.info(f"回测开始，组合{len(top)}只，数据{len(monthly)}行")

if not all(col in top.columns for col in ["symbol", "weight"]):
    raise ValueError("top 必须包含 symbol 和 weight 列")
def run_backtest(monthly: pd.DataFrame, top: pd.DataFrame):

   """回测+可视化"""
    port = top[["symbol", "weight"]].copy()
    merged = monthly.merge(port, on="symbol", how="inner")
    merged["weighted_ret"] = merged["monthly_ret"] * merged["weight"]

    port_monthly = merged.groupby("date").agg(
        monthly_ret=("weighted_ret", "sum")
    ).reset_index().sort_values("date")
    port_monthly["cum_ret"] = (1 + port_monthly["monthly_ret"]).cumprod() - 1

    # 下载标普500对比
    sp500 = yf.download("^GSPC", start=port_monthly["date"].min(), 
                        end=port_monthly["date"].max() + pd.Timedelta(days=5), progress=False)
    if not sp500.empty:
        sp500_close = sp500["Close"].squeeze()
        sp500_monthly = sp500_close.resample("ME").last()
        sp500_ret = sp500_monthly.pct_change().dropna()
        sp500_df = pd.DataFrame({
            "date": sp500_ret.index.ravel(),
            "sp500_ret": sp500_ret.values.ravel()
        })
        sp500_df["cum_sp500"] = (1 + sp500_df["sp500_ret"]).cumprod() - 1
    else:
        sp500_df = pd.DataFrame(columns=["date", "sp500_ret", "cum_sp500"])

    # 绘图
    merged_plot = port_monthly.merge(sp500_df, on="date", how="left")
    
    plt.figure(figsize=(10, 5))
    plt.plot(merged_plot["date"], merged_plot["cum_ret"], marker="o", label="多因子组合")
    if not sp500_df.empty:
        plt.plot(merged_plot["date"], merged_plot["cum_sp500"], linestyle="-.", label="标普500")
    plt.title("Portfolio Cumulative Return vs S&P 500")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "sample_demo_cum_return.png", dpi=150)
    plt.close()

    # 保存
    merged_plot.to_csv(OUTPUTS_DIR / "sample_demo_backtest.csv", index=False)
    return merged_plot