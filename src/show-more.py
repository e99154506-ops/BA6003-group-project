import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf

# ================= 0. 页面配置 =================
st.set_page_config(page_title="BA6003 Multi-Factor Dashboard", layout="wide")
st.title("📈 Multi-Factor Stock Selection & Backtest")

# ================= 1. 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ================= 2. 真实 S&P 500 数据加载（防崩兜底） =================
@st.cache_data(show_spinner="Fetching real S&P 500 (^GSPC) data...")
def load_sp500_real(start="2023-01-01", end="2023-12-31"):
    """
    真实拉取 S&P 500 (^GSPC)
    返回: DataFrame[date, sp500_ret, sp500_cum]
    """
    try:
        raw = yf.download("^GSPC", start=start, end=end, progress=False)

        # yfinance 新版会返回 MultiIndex columns，这里压平
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(-1)

        close = raw["Close"]
        close = pd.Series(close).squeeze()  # 防多维报错

        monthly_close = close.resample("ME").last()
        sp500_ret = monthly_close.pct_change().dropna()

        sp500_df = sp500_ret.rename("sp500_ret").to_frame()
        sp500_df["sp500_cum"] = (1 + sp500_df["sp500_ret"]).cumprod() - 1
        sp500_df = sp500_df.reset_index().rename(columns={"Date": "date"})

        return sp500_df, "yfinance: ^GSPC"

    except Exception as e:
        # 回退：模拟一条 ~24% 年收益的 S&P 500
        dates = pd.date_range(start="2023-01-31", end="2023-12-31", freq="ME")
        rng = np.random.default_rng(2023)
        rets = rng.normal(0.02, 0.04, len(dates))
        sp500_df = pd.DataFrame({"date": dates, "sp500_ret": rets})
        sp500_df["sp500_cum"] = (1 + sp500_df["sp500_ret"]).cumprod() - 1

        return sp500_df, f"fallback simulated (yfinance failed: {e})"


# ================= 3. 侧边栏互动参数 =================
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    
    # 数据源选择（本地 CSV 为默认）
    data_source = st.radio(
        "Data Source",
        ["Local CSV (Offline)", "Simulated (Demo)"],
        index=0,
        help="Local CSV uses data/stock_monthly_returns_sample.csv"
    )
    
    # 选股数量（本地数据只有4只，限制1-4）
    top_n = st.slider("Top N Holdings", 1, 4, 3)
    
    st.subheader("Weight Formula: Alpha - k*Beta")
    alpha_w = st.slider("Alpha Weight", 0.0, 1.0, 0.6, 0.1)
    beta_w = st.slider("Beta Weight", 0.0, 1.0, 0.4, 0.1)
    
    # 图表开关
    show_sp500 = st.checkbox("Show S&P 500 Benchmark", True)
    show_bar = st.checkbox("Show Monthly Returns Bar", True)
    
    st.markdown("---")
    st.caption("BA6003 Group Project Demo")


# ================= 4. 数据加载（防崩兜底） =================
@st.cache_data
def load_monthly_returns(source):
    """加载月度收益数据"""
    if source == "Local CSV (Offline)":
        csv_path = os.path.join(DATA_DIR, "stock_monthly_returns_sample.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                # 校验必需列
                if not all(col in df.columns for col in ["symbol", "date", "monthly_ret"]):
                    st.warning("CSV missing required columns, falling back to simulated data")
                    return generate_simulated_monthly()
                
                # 校验收益列是否为小数（0.0x），若为价格绝对值（如 150.0）则自动转换
                sample = df["monthly_ret"].dropna().iloc[0]
                if abs(sample) > 1.0:  # 大于 100% 视为价格，需转收益率
                    st.warning("Detected price-like values in 'monthly_ret', converting to returns...")
                    # 按 symbol 分组，计算 pct_change
                    df = df.sort_values(["symbol", "date"])
                    df["monthly_ret"] = df.groupby("symbol")["monthly_ret"].pct_change()
                    df = df.dropna(subset=["monthly_ret"])
                
                return df
            except Exception as e:
                st.warning(f"Error reading CSV: {e}, falling back to simulated data")
                return generate_simulated_monthly()
        else:
            st.warning("Local CSV not found, falling back to simulated data")
            return generate_simulated_monthly()
    else:
        return generate_simulated_monthly()


def generate_simulated_monthly():
    """生成模拟月度收益（防崩兜底）"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-02-28', periods=12, freq='ME')
    symbols = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
    data = []
    for s in symbols:
        for d in dates:
            data.append([s, d, np.random.randn()*0.05 + 0.02])
    return pd.DataFrame(data, columns=['symbol', 'date', 'monthly_ret'])


# 加载数据
monthly = load_monthly_returns(data_source)


# ================= 5. 因子计算（基于真实月度收益） =================
def compute_factors(monthly_df):
    """从月度收益计算 Alpha 和 Beta"""
    # 市场收益 = 所有股票月度收益的均值
    market_ret = monthly_df.groupby("date")["monthly_ret"].mean().rename("mkt_ret")
    
    # 合并个股与市场收益
    merged = monthly_df.merge(market_ret, on="date")
    
    factors = []
    for symbol in merged["symbol"].unique():
        stock = merged[merged["symbol"] == symbol]
        # Beta = Cov(stock, mkt) / Var(mkt)
        var_mkt = np.var(stock["mkt_ret"])
        beta = np.cov(stock["monthly_ret"], stock["mkt_ret"])[0, 1] / var_mkt if var_mkt > 0 else 1.0
        # Alpha = 平均超额收益
        alpha = stock["monthly_ret"].mean() - beta * stock["mkt_ret"].mean()
        factors.append({"symbol": symbol, "alpha": alpha, "beta_mkt": beta})
    
    return pd.DataFrame(factors)


factor_df = compute_factors(monthly)


# ================= 6. 选股逻辑（互动参数生效） =================
factor_df["score"] = alpha_w * factor_df["alpha"] - beta_w * factor_df["beta_mkt"]
factor_df = factor_df.sort_values("score", ascending=False)

top = factor_df.head(top_n).copy()
# 权重 = 得分归一化（可多空）
score_sum = top["score"].abs().sum()
if score_sum > 0:
    top["weight"] = top["score"] / score_sum
else:
    top["weight"] = 1.0 / len(top)  # 防除零


# ================= 7. 组合收益计算 =================
# 合并权重与月度收益
merged = monthly.merge(top[["symbol", "weight"]], on="symbol", how="inner")
merged["weighted_ret"] = merged["monthly_ret"] * merged["weight"]

# 组合月度收益
port_monthly = merged.groupby("date").agg(
    monthly_ret=("weighted_ret", "sum")
).reset_index().sort_values("date")

# 防极端值：单月收益限制在 -20% 到 20% 之间（真实股票极少超此范围）
port_monthly["monthly_ret"] = port_monthly["monthly_ret"].clip(-0.2, 0.2)
port_monthly["cum_ret"] = (1 + port_monthly["monthly_ret"]).cumprod() - 1


# ================= 8. 加载真实 S&P 500 并合并 =================
sp500_df, sp500_source = load_sp500_real("2023-01-01", "2023-12-31")
st.caption(f"S&P 500 source: {sp500_source}")

# 和组合月度日期对齐（只 merge 一次！）
merged_plot = port_monthly.merge(sp500_df, on="date", how="left")

# 如果某些月份对不上，向前填充
merged_plot["sp500_cum"] = merged_plot["sp500_cum"].ffill()
merged_plot["sp500_ret"] = merged_plot["sp500_ret"].ffill()


# ================= 9. KPI 指标 =================
st.header("Core Indicators (KPIs)")
final_ret = merged_plot["cum_ret"].iloc[-1]*100 if not merged_plot.empty else 0
sp500_ret = merged_plot["sp500_cum"].iloc[-1]*100 if not merged_plot.empty else 0
excess = final_ret - sp500_ret

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Portfolio Cumulative Return", f"{final_ret:.2f}%")
with col2: st.metric("S&P 500 Cumulative Return", f"{sp500_ret:.2f}%")
with col3: st.metric("Excess Return (Alpha)", f"{excess:.2f}%", delta=f"{excess:.2f}%")
with col4: st.metric("Holdings Count", f"{len(top)}")


# ================= 10. 选股结果展示 =================
st.subheader("🏆 Stock Selection & Weights")
col_t, col_w = st.columns([1, 1])

with col_t:
    st.dataframe(top[["symbol", "alpha", "beta_mkt", "score", "weight"]], use_container_width=True)

with col_w:
    fig_w, ax_w = plt.subplots(figsize=(6, 3))
    colors_w = ["green" if w >= 0 else "red" for w in top["weight"]]
    ax_w.bar(top["symbol"], top["weight"], color=colors_w, alpha=0.7)
    ax_w.axhline(0, color="black", alpha=0.5)
    ax_w.set_title("Weight Distribution")
    ax_w.set_ylabel("Weight")
    ax_w.grid(axis="y", alpha=0.3)
    st.pyplot(fig_w)


# ================= 11. 收益走势图 =================
st.subheader("📈 Cumulative Return Comparison")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(merged_plot["date"], merged_plot["cum_ret"], marker="o", color="#1f77b4", linewidth=2, label="Multi-Factor Portfolio")
if show_sp500 and "sp500_cum" in merged_plot.columns:
    ax1.plot(merged_plot["date"], merged_plot["sp500_cum"], linestyle="-.", color="#2ca02c", alpha=0.8, label="S&P 500")
ax1.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax1.set_ylabel("Cumulative Return")
ax1.grid(alpha=0.3)
ax1.legend()
st.pyplot(fig1)


# ================= 12. 月度收益分布图（修正版） =================
if show_bar:
    st.subheader("📊 Monthly Return Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    colors = ["green" if x >= 0 else "red" for x in merged_plot["monthly_ret"]]
    ax2.bar(merged_plot["date"], merged_plot["monthly_ret"], width=20, color=colors, alpha=0.7, label="Portfolio")
    
    # 叠加 S&P 500 月度收益
    if show_sp500:
        ax2.plot(merged_plot["date"], merged_plot["sp500_ret"], marker="o", linestyle="--", color="#2ca02c", alpha=0.6, label="S&P 500")
    
    ax2.axhline(0, color="black", alpha=0.5)
    ax2.set_title("Monthly Return Distribution")
    ax2.set_ylabel("Monthly Return")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax2.set_ylim(-0.25, 0.25)  # 固定 Y 轴防极端值拉伸
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)


# ================= 13. 明细数据 =================
with st.expander("🔍 View Backtest Details"):
    st.dataframe(merged_plot.sort_values("date").style.format({
        "monthly_ret": "{:.2%}", "cum_ret": "{:.2%}",
        "sp500_ret": "{:.2%}", "sp500_cum": "{:.2%}"
    }))

st.caption(f"Data Source: {data_source} | Alpha Weight: {alpha_w} | Beta Weight: {beta_w}")