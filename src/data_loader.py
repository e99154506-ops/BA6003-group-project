# src/data_loader.py
import yfinance as yf
import pandas as pd
from .config import TICKERS, START_DATE, END_DATE, OUTPUTS_DIR

def download_prices() -> pd.DataFrame:
    """下载股票日线数据，返回长表"""
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by="ticker")
    prices = []
    for t in TICKERS:
        df = data[t].copy().reset_index()
        df["symbol"] = t
        prices.append(df[["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]])
    result = pd.concat(prices).rename(columns={"Date": "date", "Close": "close"})
    result["date"] = pd.to_datetime(result["date"])
    return result

def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """从日线计算月度收益率"""
    prices_idx = prices.set_index("date")
    monthly_ret = (
        prices_idx
        .groupby("symbol")["close"]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
        .rename("monthly_ret")
        .reset_index()
    )
    return monthly_ret

def save_outputs(prices: pd.DataFrame, monthly_ret: pd.DataFrame):
    """保存到 outputs/"""
    prices.to_csv(OUTPUTS_DIR / "stock_prices_sample.csv", index=False)
    monthly_ret.to_csv(OUTPUTS_DIR / "stock_monthly_returns_sample.csv", index=False)