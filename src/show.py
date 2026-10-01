import pandas as pd
import numpy as np
import os

# 1. 路径设置
desktop = "/Users/kwich/Desktop"
prices_path = os.path.join(desktop, "stock_prices_sample.csv")
monthly_path = os.path.join(desktop, "stock_monthly_returns_sample.csv")

# 2. 加载样本数据
print("📥 加载样本数据...")
prices = pd.read_csv(prices_path, parse_dates=["date"])
monthly = pd.read_csv(monthly_path, parse_dates=["date"])

print(f"日线数据: {len(prices)} 行, 股票: {prices['symbol'].nunique()} 只")
print(f"月度收益: {len(monthly)} 行")

# 3. 构造简易 FF3 因子（样本演示用，实际应替换为你下载的真实 FF3）
# 这里用随机模拟 Mkt-RF, SMB, HML, RF，仅作演示逻辑
dates = monthly["date"].unique()
np.random.seed(42)
ff3_factors = pd.DataFrame({
    "date": dates,
    "Mkt-RF": np.random.normal(0.008, 0.04, len(dates)),
    "SMB": np.random.normal(0.002, 0.02, len(dates)),
    "HML": np.random.normal(0.001, 0.02, len(dates)),
    "RF": [0.0003]*len(dates)  # 无风险利率假设
})
ff3_factors["date"] = pd.to_datetime(ff3_factors["date"])

# 4. 合并月度收益与因子（为每只股票准备回归数据）
demo_data = monthly.merge(ff3_factors, on="date", how="left")
demo_data["excess_ret"] = demo_data["monthly_ret"] - demo_data["RF"]

# 5. 简易因子回归（每只股票对 FF3 做 OLS，取最近期结果作演示）
print("\n📊 进行简易 FF3 因子回归（样本演示）...")
results = []
for symbol in demo_data["symbol"].unique():
    df = demo_data[demo_data["symbol"] == symbol].dropna()
    if len(df) < 10:
        continue
    # Y = excess_ret, X = Mkt-RF, SMB, HML
    X = df[["Mkt-RF", "SMB", "HML"]]
    X = np.column_stack([np.ones(len(X)), X])  # 加截距
    Y = df["excess_ret"].values
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        results.append({
            "symbol": symbol,
            "alpha": beta[0],
            "beta_mkt": beta[1],
            "beta_smb": beta[2],
            "beta_hml": beta[3],
            "recent_ret": df["monthly_ret"].iloc[-1]
        })
    except Exception:
        continue

factor_df = pd.DataFrame(results)
print(f"回归完成: {len(factor_df)} 只股票")

# 6. 模拟组合构建（演示：低波动+高Alpha）
# 权重：Alpha 越高、市场 Beta 越低，权重越大
factor_df["score"] = factor_df["alpha"] * 100 - factor_df["beta_mkt"] * 0.2
top = factor_df.sort_values("score", ascending=False).head(3).copy()
top["weight"] = top["score"] / top["score"].sum()

# 7. 输出演示结果
print("\n🏆 样本选股演示结果（Top 3）:")
print(top[["symbol", "alpha", "beta_mkt", "score", "weight"]].to_string(index=False))

# 保存演示结果
demo_out = os.path.join(desktop, "sample_demo_portfolio.csv")
top.to_csv(demo_out, index=False)
print(f"\n✅ 演示组合已保存: {demo_out}")

# 8. 简单可视化（可选，存为图片）
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.bar(top["symbol"], top["weight"])
    plt.title("Sample Portfolio Weights (Demo)")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(desktop, "sample_demo_weights.png"))
    print("📈 权重图已保存: sample_demo_weights.png")
except ImportError:
    print("（matplotlib 未安装，跳过绘图）")

# ================= 9. 选股后组合收益率可视化 =================
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # 读取原始月度收益（确保路径正确，当前演示用桌面路径）
    monthly_path = "/Users/kwich/Desktop/stock_monthly_returns_sample.csv"
    # 若已统一到 data 目录，改用：monthly_path = os.path.join(data_dir, "stock_monthly_returns_sample.csv")

    if os.path.exists(monthly_path):
        monthly_ret = pd.read_csv(monthly_path)
        monthly_ret["date"] = pd.to_datetime(monthly_ret["date"])

        # 合并权重（top 是上一步算出的 Top 3 组合）
        port = top[["symbol", "weight"]].rename(columns={"symbol": "symbol"})
        merged = monthly_ret.merge(port, left_on="symbol", right_on="symbol", how="inner")

        # 计算每月加权收益
        merged["weighted_ret"] = merged["monthly_ret"] * merged["weight"]

        # 按月份汇总组合整体收益
        port_monthly = merged.groupby("date").agg(
            monthly_ret=("weighted_ret", "sum"),
            symbols=("symbol", lambda x: ",".join(x))
        ).reset_index().sort_values("date")

        # 计算累计收益（假设初始本金 1）
        port_monthly["cum_ret"] = (1 + port_monthly["monthly_ret"]).cumprod() - 1

        # --- 图1：累计收益曲线 ---
        plt.figure(figsize=(8, 4))
        plt.plot(port_monthly["date"], port_monthly["cum_ret"], marker="o", label="Portfolio Cumulative Return")
        plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
        plt.title("Selected Portfolio Cumulative Return (Demo)")
        plt.ylabel("Cumulative Return")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        cum_path = "/Users/kwich/Desktop/sample_demo_cum_return.png"
        plt.savefig(cum_path)
        print(f"📈 累计收益图已保存: {cum_path}")

        # --- 图2：月度收益柱状图（红绿区分涨跌） ---
        plt.figure(figsize=(8, 4))
        colors = ["green" if x >= 0 else "red" for x in port_monthly["monthly_ret"]]
        plt.bar(port_monthly["date"], port_monthly["monthly_ret"], color=colors, alpha=0.7)
        plt.axhline(0, color="black", alpha=0.5)
        plt.title("Selected Portfolio Monthly Return (Demo)")
        plt.ylabel("Monthly Return")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        monthly_path_out = "/Users/kwich/Desktop/sample_demo_monthly_return.png"
        plt.savefig(monthly_path_out)
        print(f"📊 月度收益图已保存: {monthly_path_out}")

        # 保存回测结果 CSV
        port_monthly.to_csv("/Users/kwich/Desktop/sample_demo_backtest.csv", index=False)
        print("✅ 组合回测数据已保存: sample_demo_backtest.csv")

    else:
        print("（未找到月度收益文件，跳过收益图）")

except ImportError:
    print("（matplotlib/pandas 未安装，跳过收益图）")