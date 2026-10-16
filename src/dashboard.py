from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import os
import requests
from dotenv import load_dotenv
import altair as alt

# 确保从项目根目录启动时可以导入 src 包
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / "src" / ".env")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import PRICE_CSV
from src.data_loader import compute_monthly_returns, load_prices


LOOKBACK = 6
RISK_FREE_RATE = 0.0003


# ================= 数据加载 =================
@st.cache_data
def load_dashboard_data():
    """读取原始价格数据，并通过项目数据层计算月度收益。"""
    prices = load_prices()
    monthly_returns, _ = compute_monthly_returns(prices)

    monthly = (
        monthly_returns
        .rename_axis(index="date", columns="symbol")
        .stack()
        .rename("monthly_ret")
        .reset_index()
    )
    monthly["symbol"] = monthly["symbol"].astype(str)
    monthly["monthly_ret"] = pd.to_numeric(
        monthly["monthly_ret"], errors="coerce"
    )
    monthly = monthly.dropna(subset=["date", "symbol", "monthly_ret"])
    monthly = monthly.sort_values(["symbol", "date"])

    return monthly, prices


# ================= 因子计算 =================
def build_ff3_factors(monthly_df, risk_free_rate=RISK_FREE_RATE):
    """Demonstration factors; replace with real FF3 data for production."""
    dates = pd.Series(monthly_df["date"].dropna().unique()).sort_values()
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": rng.normal(0.008, 0.04, len(dates)),
            "SMB": rng.normal(0.002, 0.02, len(dates)),
            "HML": rng.normal(0.001, 0.02, len(dates)),
            "RF": risk_free_rate,
        }
    )


def calc_momentum(monthly_df, lookback=LOOKBACK):
    df = monthly_df.sort_values(["symbol", "date"]).copy()
    df["mom"] = df.groupby("symbol")["monthly_ret"].transform(
        lambda values: values.rolling(lookback, min_periods=lookback).mean()
    )
    return df[["symbol", "date", "mom"]].dropna()


def calc_ff3_betas(monthly_df, ff3):
    merged = monthly_df.merge(ff3, on="date", how="left")
    merged["excess_ret"] = merged["monthly_ret"] - merged["RF"]

    results = []
    for symbol, data in merged.groupby("symbol"):
        data = data.dropna(subset=["excess_ret", "Mkt-RF", "SMB", "HML"])
        if len(data) < 10:
            continue

        x = np.column_stack(
            [
                np.ones(len(data)),
                data["Mkt-RF"],
                data["SMB"],
                data["HML"],
            ]
        )
        beta, *_ = np.linalg.lstsq(
            x, data["excess_ret"].to_numpy(), rcond=None
        )
        results.append(
            {
                "symbol": symbol,
                "alpha": beta[0],
                "beta_mkt": beta[1],
                "beta_smb": beta[2],
                "beta_hml": beta[3],
            }
        )

    return pd.DataFrame(results)


def calculate_factors(monthly_df, lookback=LOOKBACK, risk_free_rate=RISK_FREE_RATE):
    momentum = calc_momentum(monthly_df, lookback=lookback)
    if momentum.empty:
        return pd.DataFrame()

    latest_momentum = (
        momentum.sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
    )

    ff3 = build_ff3_factors(monthly_df, risk_free_rate=risk_free_rate)
    betas = calc_ff3_betas(monthly_df, ff3)
    if betas.empty:
        return pd.DataFrame()

    combined = latest_momentum.merge(betas, on="symbol", how="inner")
    combined["score"] = (
        combined["mom"] * 100
        + combined["alpha"] * 100
        - combined["beta_mkt"] * 0.2
    )

    latest_returns = (
        monthly_df.sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "monthly_ret"]]
        .rename(columns={"monthly_ret": "recent_ret"})
    )
    return combined.merge(latest_returns, on="symbol", how="left")


def performance_summary(returns, risk_free_rate=RISK_FREE_RATE):
    returns = returns.dropna()
    if returns.empty:
        return {}

    nav = (1 + returns).cumprod()
    periods = len(returns)
    annual_return = (
        nav.iloc[-1] ** (12 / periods) - 1
        if nav.iloc[-1] > 0
        else np.nan
    )
    annual_volatility = returns.std() * np.sqrt(12)
    sharpe = (
        (returns.mean() - risk_free_rate) * 12 / annual_volatility
        if annual_volatility > 0
        else 0
    )

    nav_with_initial = pd.concat(
        [
            pd.Series(
                [1.0],
                index=[returns.index[0] - pd.Timedelta(days=1)],
            ),
            nav,
        ]
    )
    drawdown = nav_with_initial / nav_with_initial.cummax() - 1

    return {
        "Total Return": nav.iloc[-1] - 1,
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": drawdown.min(),
    }


# ================= Streamlit 页面 =================
def main():
    st.set_page_config(
        page_title="Stock Strategy Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Stock Strategy Dashboard")
    st.caption(
        "Monthly returns are calculated from raw price data. "
        "Explore momentum and FF3-based stock selection."
    )

    try:
        monthly, prices = load_dashboard_data()
    except Exception as exc:
        st.error(f"Failed to load raw prices or calculate monthly returns: {exc}")
        st.info(f"Check that PRICE_CSV in config.py points to the correct file: {PRICE_CSV}")
        st.stop()

    if monthly.empty:
        st.error("No monthly return data is available. Check the raw prices CSV.")
        st.stop()

    st.sidebar.header("Strategy Controls")

    max_top_n = max(1, monthly["symbol"].nunique())
    top_n = st.sidebar.slider(
        "Number of stocks",
        min_value=1,
        max_value=max_top_n,
        value=min(3, max_top_n),
    )

    lookback = st.sidebar.slider(
        "Momentum lookback (months)",
        min_value=3,
        max_value=12,
        value=LOOKBACK,
        help="Number of past monthly returns used to calculate momentum.",
    )

    annual_risk_free_rate = st.sidebar.slider(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.36,
        step=0.1,
    )
    monthly_risk_free_rate = annual_risk_free_rate / 100 / 12

    show_monthly_chart = st.sidebar.checkbox(
        "Show monthly returns chart",
        value=True,
    )
    show_raw_data = st.sidebar.checkbox(
        "Show raw price data",
        value=False,
    )

    factors = calculate_factors(
        monthly,
        lookback=lookback,
        risk_free_rate=monthly_risk_free_rate,
    )
    if factors.empty:
        st.error(
            "There is not enough historical data to calculate the factors. "
            "Check the data and make sure each stock has at least 10 months of history."
        )
        st.stop()

    # Select the top-scoring stocks and assign portfolio weights.
    portfolio = factors.nlargest(top_n, "score").copy()
    shifted_scores = portfolio["score"] - portfolio["score"].min()

    if shifted_scores.sum() > 0:
        portfolio["weight"] = shifted_scores / shifted_scores.sum()
    else:
        portfolio["weight"] = 1.0 / len(portfolio)

    # Calculate weighted monthly portfolio returns.
    weighted_returns = monthly.merge(
        portfolio[["symbol", "weight"]],
        on="symbol",
        how="inner",
    )
    weighted_returns["weighted_ret"] = (
        weighted_returns["monthly_ret"] * weighted_returns["weight"]
    )
    returns = (
        weighted_returns.groupby("date")["weighted_ret"]
        .sum()
        .sort_index()
        .rename("monthly_ret")
    )

    stats = performance_summary(
        returns,
        risk_free_rate=monthly_risk_free_rate,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Stocks", f"{monthly['symbol'].nunique():,}")
    col2.metric("Raw Price Rows", f"{len(prices):,}")
    col3.metric(
        "Data Period",
        f"{monthly['date'].min():%Y-%m} to {monthly['date'].max():%Y-%m}",
    )

    st.subheader("Backtest Performance")
    if stats:
        metric_cols = st.columns(len(stats))
        for column, (name, value) in zip(metric_cols, stats.items()):
            if name == "Sharpe Ratio":
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:.2%}"
            column.metric(name, formatted)
    else:
        st.info("There is not enough return data to calculate performance metrics.")

    if not returns.empty:
        nav = (1 + returns).cumprod()
        st.subheader("Strategy Cumulative Net Worth")

        nav_chart = pd.DataFrame({
            "Date": pd.to_datetime(nav.index),
            "Strategy Value": nav.to_numpy(),
        })

        chart = (
            alt.Chart(nav_chart)
            .mark_line()
            .encode(
                x=alt.X(
                    "Date:T",
                    axis=alt.Axis(
                        format="%Y-%m",
                        tickCount=10,
                        labelAngle=-45,
                        title="Year-Month",
                    ),
                ),
                y=alt.Y(
                    "Strategy Value:Q",
                    scale=alt.Scale(zero=False),
                    title="Cumulative Value",
                ),
                tooltip=[
                    alt.Tooltip("Date:T", format="%Y-%m"),
                    alt.Tooltip("Strategy Value:Q", format=".2f"),
                ],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Monthly Returns")
        st.bar_chart(returns.rename("Monthly Return"))

    st.subheader(f"Selected Stocks (Top {len(portfolio)})")
    display_columns = [
        column
        for column in [
            "symbol",
            "score",
            "mom",
            "alpha",
            "beta_mkt",
            "recent_ret",
            "weight",
        ]
        if column in portfolio.columns
    ]
    st.dataframe(
        portfolio[display_columns].sort_values("score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if show_raw_data:
        with st.expander("Raw Price Data", expanded=True):
            st.dataframe(prices, use_container_width=True, hide_index=True)

    with st.expander("Data Sources"):
        st.write(f"Raw price data: `{PRICE_CSV}`")
        st.write("Monthly returns are calculated from raw prices by `src/data_loader.py`.")
        st.warning(
            "FF3 factors are currently randomly generated for demonstration, "
            "so alpha and beta results are not reliable. "
            "The backtest may also contain look-ahead bias and should not be "
            "treated as investment advice."
        )

    st.subheader("🤖 AI Insight")

    if st.button("Generate AI Strategy Insight", type="primary"):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            st.error("OPENROUTER_API_KEY was not found. Check src/.env.")
        elif not stats:
            st.warning("There is not enough backtest data to analyze.")
        else:
            summary = {
                "performance": {
                    name: round(float(value), 4)
                    for name, value in stats.items()
                    if pd.notna(value)
                },
                "selected_stocks": portfolio[
                    [
                        col for col in
                        ["symbol", "weight", "score", "mom", "alpha", "beta_mkt"]
                        if col in portfolio.columns
                    ]
                ].round(4).to_dict(orient="records"),
                "period": (
                    f"{monthly['date'].min():%Y-%m} to "
                    f"{monthly['date'].max():%Y-%m}"
                ),
            }

            prompt = f"""
Analyze the following strategy backtest results in English. Explain what the
returns, volatility, Sharpe ratio, and maximum drawdown indicate. Briefly
comment on the selected stocks. Clearly state the limitations and do not give
investment advice.

Results:
{summary}

Important limitations: FF3 factors in this dashboard are randomly generated
demonstration data, so alpha and beta conclusions are unreliable. The current
backtest may also contain look-ahead bias.
"""

            with st.spinner("Analyzing the strategy results…"):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": os.getenv(
                                "OPENROUTER_MODEL",
                                "openai/gpt-4o-mini",
                            ),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a cautious, objective quantitative "
                                        "strategy analysis assistant. Always respond "
                                        "in English."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.3,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    insight = response.json()["choices"][0]["message"]["content"]
                    st.markdown(insight)
                except requests.RequestException as exc:
                    st.error(f"AI service request failed: {exc}")
                except (KeyError, IndexError, ValueError):
                    st.error("The AI service returned an unreadable response. Please try again.")


if __name__ == "__main__":
    main()