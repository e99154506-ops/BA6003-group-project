import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.ticker as mtick
import platform
import sys

# ========== 修复中文乱码 ==========
sys_name = platform.system()
if sys_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif sys_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 路径设置
desktop = "/Users/kwich/Desktop"
prices_path = os.path.join(desktop, "stock_prices_sample.csv")
monthly_path = os.path.join(desktop, "stock_monthly_returns_sample.csv")

# 2. 加载样本数据
print("📥 加载样本数据...")
if not os.path.exists(monthly_path):
    print("❌ 未找到月度收益文件，请检查桌面路径！")
    sys.exit(1)

prices = pd.read_csv(prices_path, parse_dates=["date"]) if os.path.exists(prices_path) else None
monthly = pd.read_csv(monthly_path, parse_dates=["date"])
print(f"月度收益: {len(monthly)} 行, 股票: {monthly['symbol'].nunique()} 只")

# 3. 构造简易 FF3 因子（演示用）
dates = monthly["date"].unique()
np.random.seed(42)
ff3_factors = pd.DataFrame({
    "date": dates,
    "Mkt-RF": np.random.normal(0.008, 0.04, len(dates)),
    "SMB": np.random.normal(0.002, 0.02, len(dates)),
    "HML": np.random.normal(0.001, 0.02, len(dates)),
    "RF": [0.0003] * len(dates)
})
ff3_factors["date"] = pd.to_datetime(ff3_factors["date"])

# 4. 合并因子并回归
demo_data = monthly.merge(ff3_factors, on="date", how="left")
demo_data["excess_ret"] = demo_data["monthly_ret"] - demo_data["RF"]

print("📊 进行简易 FF3 因子回归...")
results = []
for symbol in demo_data["symbol"].unique():
    df = demo_data[demo_data["symbol"] == symbol].dropna()
    if len(df) < 10: continue
    X = np.column_stack([np.ones(len(df)), df["Mkt-RF"], df["SMB"], df["HML"]])
    Y = df["excess_ret"].values
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    results.append({
        "symbol": symbol, "alpha": beta[0],
        "beta_mkt": beta[1], "recent_ret": df["monthly_ret"].iloc[-1]
    })

factor_df = pd.DataFrame(results)
if factor_df.empty:
    print("❌ 回归无结果，退出")
    sys.exit(1)

# 5. 模拟组合构建（低波动+高Alpha） -> 生成 'top'
factor_df["score"] = factor_df["alpha"] * 100 - factor_df["beta_mkt"] * 0.2
top = factor_df.sort_values("score", ascending=False).head(3).copy()
top["weight"] = top["score"] / top["score"].sum()

print("\n🏆 选股结果 (Top 3):")
print(top[["symbol", "alpha", "beta_mkt", "weight"]].to_string(index=False))

# ========== 6. 选股后组合收益率可视化（含标普500对比） ==========
print("\n📈 开始计算回测与标普500对比...")
monthly_ret = monthly.copy()
port = top[["symbol", "weight"]]  # 这里就是报错修复点，top已在上文定义
merged = monthly_ret.merge(port, on="symbol", how="inner")
merged["weighted_ret"] = merged["monthly_ret"] * merged["weight"]

# 组合月度与累计收益
port_monthly = merged.groupby("date").agg(
    monthly_ret=("weighted_ret", "sum")
).reset_index().sort_values("date")
port_monthly["cum_ret"] = (1 + port_monthly["monthly_ret"]).cumprod() - 1

# 下载标普500
# --- 下载标普500 (保持你原有的下载代码) ---
start_date = port_monthly["date"].min()
end_date = port_monthly["date"].max()
print(f"📥 下载标普500 ({start_date.date()} 至 {end_date.date()})...")
sp500 = yf.download("^GSPC", start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)

if not sp500.empty:
    # 修复点：提取 Close 并强制转为一维，兼容 yfinance 新版本
    sp500_close = sp500["Close"].squeeze()
    sp500_monthly = sp500_close.resample("ME").last()
    sp500_ret = sp500_monthly.pct_change().dropna()

    # 修复点：强制压平索引和数值，解决 ValueError
    sp500_df = pd.DataFrame({
        "date": sp500_ret.index.ravel(),
        "sp500_ret": sp500_ret.values.ravel()
    })
    sp500_df["cum_sp500"] = (1 + sp500_df["sp500_ret"]).cumprod() - 1
    print(f"✅ 标普500数据: {len(sp500_df)} 个月")
else:
    sp500_df = pd.DataFrame(columns=["date", "sp500_ret", "cum_sp500"])
    print("⚠️ 标普500下载失败，跳过对比")

# 后续你的合并与绘图代码保持不变...
merged_plot = port_monthly.merge(sp500_df, on="date", how="left")

# --- 绘图：累计收益对比 ---
plt.figure(figsize=(10, 5))
plt.plot(merged_plot["date"], merged_plot["cum_ret"], marker="o",
         color="#1f77b4", linewidth=2, label="多因子组合 (Top 3)")
if not sp500_df.empty:
    plt.plot(merged_plot["date"], merged_plot["cum_sp500"],
             linestyle="-.", color="#2ca02c", alpha=0.8, label="标普500 (^GSPC)")
plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
plt.title("Portfolio Cumulative Return vs S&P 500 (Demo)")
plt.ylabel("Cumulative Return")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
cum_path = os.path.join(desktop, "sample_demo_cum_return.png")
plt.savefig(cum_path, dpi=150)
print(f"✅ 累计收益对比图已存至桌面: {cum_path}")

# --- 绘图：月度收益柱状图 ---
plt.figure(figsize=(10, 4))
colors = ["green" if x >= 0 else "red" for x in merged_plot["monthly_ret"]]
plt.bar(merged_plot["date"], merged_plot["monthly_ret"], color=colors, alpha=0.7)
plt.axhline(0, color="black", alpha=0.5)
plt.title("Selected Portfolio Monthly Return (Demo)")
plt.ylabel("Monthly Return")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
monthly_path_out = os.path.join(desktop, "sample_demo_monthly_return.png")
plt.savefig(monthly_path_out, dpi=150)
print(f"✅ 月度收益图已存至桌面: {monthly_path_out}")

# 保存回测CSV
merged_plot.to_csv(os.path.join(desktop, "sample_demo_backtest.csv"), index=False)
print("✅ 回测明细 CSV 已存至桌面。")