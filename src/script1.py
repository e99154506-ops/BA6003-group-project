import yfinance as yf
import pandas as pd
import os

# 桌面路径
desktop = os.path.expanduser("~/Desktop")

# 1. 下载样本数据（以几只常见股票为例）
tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
data = yf.download(tickers, start="2023-01-01", end="2024-01-01", group_by="ticker")

# 转成长表：date, symbol, open, high, low, close, volume
prices = []
for t in tickers:
    df = data[t].copy().reset_index()
    df["symbol"] = t
    prices.append(df[["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]])

prices = pd.concat(prices).rename(columns={"Date": "date", "Close": "close"})
prices["date"] = pd.to_datetime(prices["date"])

# 保存日线
prices_path = os.path.join(desktop, "stock_prices_sample.csv")
prices.to_csv(prices_path, index=False)
print(f"✅ 日线数据已保存: {prices_path} ({len(prices)} 行)")

# 2. 计算月度收益（修复版）
prices_idx = prices.set_index("date")

monthly_ret = (
    prices_idx
    .groupby("symbol")["close"]
    .resample("ME")          # ✅ 不再用 on='date'，因为 date 已经是索引
    .last()
    .pct_change()
    .dropna()
    .rename("monthly_ret")
    .reset_index()
)

# 保存月度收益
monthly_path = os.path.join("/Users/kwich/Desktop", "stock_monthly_returns_sample.csv")
monthly_ret.to_csv(monthly_path, index=False)
print(f"✅ 月度收益已保存: {monthly_path} ({len(monthly_ret)} 行)")