# src/streamlit_app.py
# 仪表盘逻辑：基于项目内 data_loader / factor_model 的真实算法
# AI 按钮：调用 src/ai_insights.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 0. 页面配置 =================
st.set_page_config(page_title="BA6003 Multi-Factor Dashboard", layout="wide")
st.title("📈 Multi-Factor Stock Selection & Backtest")

# ================= 1. 数据加载（读真实 CSV） =================
DATA_PATH = "data/stock_monthly_returns_sample.csv"

@st.cache_data
def load_monthly_returns():
    """加载月度收益数据（从 data/ 目录读取真实 CSV）"""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        if all(col in df.columns for col in ["symbol", "date", "monthly_ret"]):
            return df
    st.error(f"未找到数据文件：{DATA_PATH}")
    st.stop()

monthly = load_monthly_returns()

# ================= 2. 侧边栏互动参数 =================
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    top_n = st.slider("Top N Holdings", 1, len(monthly["symbol"].unique()), 3)
    alpha_w = st.slider("Alpha Weight", 0.0, 1.0, 0.6, 0.1)
    beta_w = st.slider("Beta Weight", 0.0, 1.0, 0.4, 0.1)
    show_sp500 = st.checkbox("Show S&P 500 Benchmark", True)
    st.caption("BA6003 Group Project Demo")

# ================= 3. 因子计算（基于真实月度收益） =================
def compute_factors(monthly_df):
    """从月度收益计算 Alpha 和 Beta（与 factor_model.py 逻辑一致）"""
    market_ret = monthly_df.groupby("date")["monthly_ret"].mean().rename("mkt_ret")
    merged = monthly_df.merge(market_ret, on="date")
    factors = []
    for symbol in merged["symbol"].unique():
        stock = merged[merged["symbol"] == symbol]
        var_mkt = np.var(stock["mkt_ret"])
        beta = np.cov(stock["monthly_ret"], stock["mkt_ret"])[0, 1] / var_mkt if var_mkt > 0 else 1.0
        alpha = stock["monthly_ret"].mean() - beta * stock["mkt_ret"].mean()
        factors.append({"symbol": symbol, "alpha": alpha, "beta_mkt": beta})
    return pd.DataFrame(factors)

factor_df = compute_factors(monthly)

# ================= 4. 选股（互动参数生效） =================
factor_df["score"] = alpha_w * factor_df["alpha"] - beta_w * factor_df["beta_mkt"]
factor_df = factor_df.sort_values("score", ascending=False)
top = factor_df.head(top_n).copy()
score_sum = top["score"].abs().sum()
top["weight"] = top["score"] / score_sum if score_sum > 0 else 1.0 / len(top)

# ================= 5. 组合收益 =================
merged = monthly.merge(top[["symbol", "weight"]], on="symbol", how="inner")
merged["weighted_ret"] = merged["monthly_ret"] * merged["weight"]
port_monthly = merged.groupby("date").agg(
    monthly_ret=("weighted_ret", "sum")
).reset_index().sort_values("date")
port_monthly["monthly_ret"] = port_monthly["monthly_ret"].clip(-0.2, 0.2)
port_monthly["cum_ret"] = (1 + port_monthly["monthly_ret"]).cumprod() - 1

# ================= 6. KPI 指标 =================
st.header("Core Indicators (KPIs)")
final_ret = port_monthly["cum_ret"].iloc[-1] * 100 if not port_monthly.empty else 0
sp500_ret_val = 9.80  # 占位，实际可通过 yfinance 获取
excess = final_ret - sp500_ret_val

col1, col2, col3 = st.columns(3)
with col1: st.metric("Portfolio Cumulative Return", f"{final_ret:.2f}%")
with col2: st.metric("S&P 500 Cumulative Return", f"{sp500_ret_val:.2f}%")
with col3: st.metric("Excess Return", f"{excess:.2f}%")

# ================= 7. 选股结果 =================
st.subheader("🏆 Stock Selection & Weights")
st.dataframe(top[["symbol", "alpha", "beta_mkt", "score", "weight"]])

# ================= 8. 累计收益图 =================
st.subheader("📈 Cumulative Return")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(port_monthly["date"], port_monthly["cum_ret"], marker="o", label="Multi-Factor Portfolio")
ax1.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax1.set_ylabel("Cumulative Return")
ax1.grid(alpha=0.3)
ax1.legend()
st.pyplot(fig1)

# ================= 9. AI 洞察（Week 10 作业核心） =================
st.markdown("---")
st.header("🤖 AI 洞察报告")

if st.button("生成 AI 诊断报告", type="primary"):
    with st.spinner("AI 正在分析当前仪表盘数据..."):
        # 1. 拼装真实上下文（全部来自仪表盘已算好的变量）
        view_context = (
            f"多因子选股回测，股票池 {list(monthly['symbol'].unique())}，"
            f"选出 Top {len(top)} 组合，Alpha 权重 {alpha_w}，Beta 权重 {beta_w}"
        )
        metrics = {
            "累计收益": f"{final_ret:.2f}%",
            "标普500收益": f"{sp500_ret_val:.2f}%",
            "超额收益": f"{excess:.2f}%",
            "选股数量": f"{len(top)} 只",
            "股票池": ", ".join(top["symbol"].tolist())
        }

        # 2. 调用 AI 模块
        from ai_insights import generate_ai_insight
        report = generate_ai_insight(view_context, metrics)

        # 3. 显示
        st.markdown("### 📋 AI 洞察")
        st.info(report)

st.caption("BA6003 Group Project | Week 10 AI Integration Demo")
