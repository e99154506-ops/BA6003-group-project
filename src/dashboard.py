# src/dashboard.py
import sys
from pathlib import Path

# 确保项目根在 sys.path，避免 No module named 'src'
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_monthly, load_spy_benchmark, filter_by_date
from src.factor_model import score_factors
from src.backtest import run_rolling_backtest

# ============ 页面配置 ============
st.set_page_config(page_title="BA6003 Factor Backtest", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.15rem !important;
        letter-spacing: -0.05em;
        color: #1f2a37;
    }
    .stCaption {
        color: #667085 !important;
        font-size: 0.95rem !important;
        margin-bottom: 1.2rem !important;
    }
    .stSidebar > div {
        background: linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stNumberInput {
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
    }
    .stTabs [role="tablist"] {
        gap: 0.6rem;
        border-bottom: 1px solid rgba(15, 23, 42, 0.12);
        margin-bottom: 1rem;
    }
    .stTabs [role="tab"] {
        padding: 0.8rem 1.1rem;
        font-weight: 600;
        border-radius: 0.6rem 0.6rem 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: rgba(11, 45, 92, 0.08);
        color: #0b2d5c;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Factor Backtest Dashboard")
st.caption("Quantitative strategy research and performance review")

# ============ 侧边栏 ============
st.sidebar.header("Strategy Settings")
st.sidebar.caption("Adjust the portfolio construction and review the backtest results.")

RISK_PROFILES = {
    "Conservative": {"top_n": 10, "vol_filter": 0.35, "max_weight": 0.04, "mom_weight": 0.5, "val_weight": 0.5},
    "Balanced":     {"top_n": 15, "vol_filter": 0.45, "max_weight": 0.06, "mom_weight": 0.6, "val_weight": 0.4},
    "Aggressive":   {"top_n": 25, "vol_filter": 0.60, "max_weight": 0.08, "mom_weight": 0.7, "val_weight": 0.3},
}

profile_name = st.sidebar.selectbox(
    "Risk Profile",
    list(RISK_PROFILES.keys()),
    index=1,
    help="Choose the strategy risk posture: conservative, balanced, or aggressive.",
)
profile = RISK_PROFILES[profile_name]

years = list(range(2015, 2026))
start_year = st.sidebar.selectbox(
    "Start Year",
    years,
    index=0,
    help="First year included in the backtest window.",
)
end_year = st.sidebar.selectbox(
    "End Year",
    years,
    index=9,
    key="end_year",
    help="Last year included in the backtest window.",
)
if end_year < start_year:
    st.sidebar.error("End Year must be >= Start Year")
    st.stop()

start_date = f"{start_year}-01-01"
end_date = f"{end_year}-12-31"

top_n = st.sidebar.number_input(
    "Top N",
    1,
    50,
    int(profile["top_n"]),
    help="Number of securities selected in each rebalance.",
)
vol_filter = st.sidebar.number_input(
    "Volatility Filter (upper quantile)",
    0.0,
    2.0,
    float(profile["vol_filter"]),
    help="Filter out the most volatile names to reduce drawdown risk.",
)
max_weight = st.sidebar.number_input(
    "Max Weight Cap",
    0.01,
    1.0,
    float(profile["max_weight"]),
    help="Maximum portfolio weight assigned to any single security.",
)
mom_weight = st.sidebar.slider(
    "Momentum Weight",
    0.0,
    1.0,
    float(profile["mom_weight"]),
    step=0.05,
    help="Importance of the momentum factor in the ranking score.",
)
val_weight = st.sidebar.slider(
    "Value Weight",
    0.0,
    1.0,
    float(profile["val_weight"]),
    step=0.05,
    help="Importance of the value/low-volatility proxy in the ranking score.",
)
method = st.sidebar.selectbox(
    "Factor Method",
    ["combo", "momentum", "value"],
    index=0,
    help="Choose how the ranking score is constructed: combined, momentum-only, or value-only.",
)

# ============ 加载数据 ============
try:
    monthly = load_monthly()
    spy = load_spy_benchmark()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

monthly_f = filter_by_date(monthly, start_date, end_date)
spy_f = filter_by_date(spy.to_frame(), start_date, end_date)["SPY"]

if len(monthly_f) < 2:
    st.error("No data in selected range after filtering.")
    st.stop()

# ============ 回测 ============
score_fn = lambda df: score_factors(
    df, method=method, mom_weight=mom_weight, val_weight=val_weight
)

equity, stats = run_rolling_backtest(
    monthly_f,
    score_fn=score_fn,
    top_n=top_n,
    vol_filter=vol_filter,
    max_weight=max_weight,
    start_date=start_date,
    end_date=end_date,
)

# ============ SPY 对齐 ============
spy_aligned = spy_f.reindex(equity.index).ffill()
spy_equity = (1 + spy_aligned.pct_change().fillna(0)).cumprod()
spy_equity = spy_equity / spy_equity.iloc[0]

compare = pd.DataFrame({
    "Strategy": equity / equity.iloc[0],
    "SPY": spy_equity,
}).dropna()

# ============ Tab 布局 ============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Equity Curve", "Monthly & Yearly Returns", "Drawdown", "Latest Selection", "Relative to SPY"
])

# ---- Tab 1: 权益曲线 ----
with tab1:
    st.subheader("Equity Curve (Strategy vs SPY)")
    st.line_chart(
        compare.assign(
            Strategy=compare["Strategy"].astype(float),
            SPY=compare["SPY"].astype(float),
        ),
        color=["#0b2d5c", "#8ecae6"],
    )

    st.subheader("Performance Stats")
    def _fmt(x):
        try:
            if isinstance(x, (int, float)) and abs(x) < 100:
                return f"{x:.2%}"
            return f"{x:.2f}"
        except Exception:
            return str(x)

    df_stats = pd.DataFrame(stats, index=["Value"]).T
    if hasattr(df_stats, "map"):
        st.table(df_stats.map(_fmt))
    else:
        st.table(df_stats.applymap(_fmt))

# ---- Tab 2: 月度/年度收益 ----
with tab2:
    st.subheader("Monthly Returns")
    strat_mret = equity.pct_change().fillna(0)
    spy_mret = spy_aligned.pct_change().fillna(0)
    monthly_ret_df = pd.DataFrame({
        "Strategy": strat_mret,
        "SPY": spy_mret,
    }).dropna()
    st.bar_chart(monthly_ret_df)

    st.subheader("Yearly Return")
    def _yr_ret(s):
        if s.empty:
            return pd.Series(dtype=float)
        y = (1 + s).resample("YE").prod() - 1
        return y

    yearly_df = pd.DataFrame({
        "Strategy": _yr_ret(strat_mret),
        "SPY": _yr_ret(spy_mret),
    }).dropna()
    st.bar_chart(yearly_df)

    st.subheader("Monthly Return Table")
    monthly_display = monthly_ret_df.copy()
    monthly_display.index = monthly_display.index.strftime("%Y-%m")
    st.dataframe(monthly_display.tail(24))  # 最近24个月

# ---- Tab 3: 回撤 ----
with tab3:
    st.subheader("Drawdown (Strategy)")
    eq_norm = equity / equity.cummax()
    dd = eq_norm - 1.0
    st.area_chart(dd, color="#ff4d4f")

    st.subheader("Drawdown (SPY)")
    spy_norm = spy_equity / spy_equity.cummax()
    dd_spy = spy_norm - 1.0
    st.area_chart(dd_spy, color="#1f77b4")

# ---- Tab 4: 最新选股 ----
with tab4:
    st.subheader("Latest Rebalance — Selected Tickers & Factor Scores")
    try:
        score_latest = score_factors(
            monthly_f, method=method, mom_weight=mom_weight, val_weight=val_weight
        )
        if isinstance(score_latest, pd.Series) and not score_latest.empty:
            # 取最高分 top_n（简化版，不做波动率二次过滤，展示用）
            sel_scores = score_latest.dropna().sort_values(ascending=False).head(top_n)
            sel_df = pd.DataFrame({
                "Rank": range(1, len(sel_scores) + 1),
                "Ticker": sel_scores.index,
                "Composite Score": sel_scores.values,
                "Weight (Equal)": [1.0 / len(sel_scores)] * len(sel_scores),
            })
            st.dataframe(sel_df, use_container_width=True)

            # 得分分布
            st.subheader("Score Distribution (Latest Period)")
            score_hist = score_latest.dropna()
            hist_df = pd.DataFrame({"Score": score_hist})
            st.bar_chart(hist_df)
        else:
            st.info("Not enough data to compute latest selection.")
    except Exception as e:
        st.warning(f"Could not compute latest selection: {e}")

# ---- Tab 5: 相对 SPY ----
with tab5:
    st.subheader("Excess Return (Strategy − SPY)")
    if not monthly_ret_df.empty:
        excess = (strat_mret - spy_mret).dropna()
        cum_excess = (1 + excess).cumprod()
        cum_excess = cum_excess / cum_excess.iloc[0]
        st.line_chart(cum_excess.to_frame("Cumulative Excess Return"), color="#0b2d5c")

        ann_excess = (1 + excess).prod() ** (12 / max(len(excess), 1)) - 1
        tracking_err = excess.std(ddof=0) * np.sqrt(12)
        info_ratio = ann_excess / tracking_err if tracking_err > 0 else 0

        rel_df = pd.DataFrame({
            "Annualized Excess Return": [ann_excess],
            "Tracking Error": [tracking_err],
            "Information Ratio": [info_ratio],
            "Strategy Total Return": [equity.iloc[-1] / equity.iloc[0] - 1],
            "SPY Total Return": [(spy_equity.iloc[-1] / spy_equity.iloc[0]) - 1],
            "Win Rate (vs SPY, monthly)": [(strat_mret > spy_mret).sum() / max(len(strat_mret.dropna()), 1)],
        })
        rel_df = rel_df.T.rename(columns={0: "Value"})

        def _fmt_rel(x):
            try:
                if isinstance(x, (int, float)) and abs(x) < 100:
                    return f"{x:.2%}"
                return f"{x:.2f}"
            except Exception:
                return str(x)

        if hasattr(rel_df, "map"):
            st.table(rel_df.map(_fmt_rel))
        else:
            st.table(rel_df.applymap(_fmt_rel))
    else:
        st.info("Not enough data for relative metrics.")

# ============ 折叠区：参数 + 免责 ============
with st.expander("Config & Data Disclaimer"):
    st.json({
        "profile": profile_name,
        "start": start_date,
        "end": end_date,
        "top_n": top_n,
        "vol_filter": vol_filter,
        "max_weight": max_weight,
        "method": method,
        "mom_weight": mom_weight,
        "val_weight": val_weight,
    })
    st.caption(
        "Using mock data: data/raw/prices_monthly.csv + spy_monthly.csv. "
        "Run `python -m src.generate_mock_data` from project root to regenerate. "
        "Demo only: validates factor/backtest pipeline & dashboard flow, not real trading returns."
    )

# ============ AI Insight ============
st.markdown("---")
st.subheader("AI Insight")

ai_models = ["deepseek/deepseek-chat", "openai/gpt-4o-mini"]
selected_model = st.selectbox(
    "AI Model",
    ai_models,
    index=0,
    help="Choose the model exposed through OpenRouter for the AI interpretation.",
)

latest_payload = {"tickers": []}
try:
    sl = score_factors(monthly_f, method=method, mom_weight=mom_weight, val_weight=val_weight)
    if isinstance(sl, pd.Series) and not sl.empty:
        latest_payload["tickers"] = list(sl.dropna().sort_values(ascending=False).head(top_n).index)
except Exception:
    pass

if st.button("Generate AI Interpretation", type="primary"):
    with st.spinner("Asking the model..."):
        try:
            from src.ai_insight import generate_insight
            text = generate_insight(
                profile=profile_name,
                start=start_date,
                end=end_date,
                stats=stats,
                latest=latest_payload,
                yearly=yearly_df.to_dict(),
                model=selected_model,
            )
            st.success("Analysis ready")
            st.markdown(text)
        except Exception as e:
            st.error(f"Could not generate insight: {e}")
            st.info("Check that OPENROUTER_API_KEY is set in .env and has remaining credits.")