# src/backtest.py
import numpy as np
import pandas as pd


def build_weights_from_selection(selected_stocks, monthly_returns, start_months: int = 6) -> pd.DataFrame:
    """Create a weight matrix with zero weights for the initial burn-in period.

    This keeps compatibility with the legacy project API that expects a weight
    DataFrame indexed by the monthly return dates.
    """
    monthly_returns = pd.DataFrame(monthly_returns).copy()
    if isinstance(selected_stocks, pd.DataFrame):
        if "symbol" in selected_stocks.columns:
            selected = selected_stocks["symbol"].dropna().astype(str).tolist()
        else:
            selected = selected_stocks.iloc[:, 0].dropna().astype(str).tolist()
    else:
        selected = list(selected_stocks)

    selected = pd.Index(pd.Series(selected).dropna().astype(str).unique())
    weights = pd.DataFrame(0.0, index=monthly_returns.index, columns=monthly_returns.columns)
    if selected.empty:
        return weights

    for i in range(start_months, len(monthly_returns.index)):
        chosen = monthly_returns.columns.intersection(selected)
        if chosen.empty:
            continue
        weight = 1.0 / len(chosen)
        weights.loc[weights.index[i], chosen] = weight
    return weights


def run_backtest(weights: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.Series:
    """Backtest portfolio returns using the previous month's weights."""
    weights = pd.DataFrame(weights).copy()
    monthly_returns = pd.DataFrame(monthly_returns).copy()

    if weights.index.equals(monthly_returns.index):
        aligned_weights = weights
    else:
        aligned_weights = weights.reindex(monthly_returns.index).ffill().fillna(0.0)

    out = pd.Series(0.0, index=monthly_returns.index, dtype=float)
    for i in range(1, len(monthly_returns.index)):
        prev_w = aligned_weights.iloc[i - 1].reindex(monthly_returns.columns).fillna(0.0)
        curr_r = monthly_returns.iloc[i].reindex(monthly_returns.columns).fillna(0.0)
        out.iloc[i] = float((prev_w * curr_r).sum())
    return out


def select_top(
    score: pd.Series,
    top_n: int = 15,
    vol_filter: float = 0.45,
    vol: pd.Series | None = None,
) -> pd.Index:
    """
    score: 截面得分 (index=tickers)
    vol: 截面波动率 (index=tickers)，如提供则做低波动过滤
    返回选中的 ticker index
    """
    score = pd.Series(score).dropna()
    if vol is not None:
        vol = pd.Series(vol).dropna()
        # 只在共同 ticker 上过滤
        common = score.index.intersection(vol.index)
        score = score.loc[common]
        vol = vol.loc[common]
        eligible = vol[vol < vol_filter].index
        score = score.loc[score.index.isin(eligible)]
    if score.empty:
        return pd.Index([])
    # 取最高分 top_n
    sel = score.sort_values(ascending=False).head(top_n).index
    return sel

def _compute_rolling_vol(prices: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """基于月频价格算月收益，再滚动std（截面每期返回Series）"""
    ret = prices.pct_change().fillna(0.0)
    # 滚动是在时间轴滚，得到每期（行）的截面波动（列=ticker）
    vol_panel = ret.rolling(window, min_periods=window).std().fillna(0.0)
    return vol_panel

def run_rolling_backtest(
    monthly_prices: pd.DataFrame,
    score_fn=None,
    top_n: int = 15,
    vol_filter: float = 0.45,
    max_weight: float = 0.06,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    rebalance_freq: int = 1,  # 月频=1
    lookback: int = 12,
) -> tuple[pd.Series, dict]:
    prices = monthly_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    mask = (prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))
    prices = prices.loc[mask].sort_index()

    if len(prices) < lookback + 1:
        return pd.Series([1.0], index=prices.index[:1]), {}

    ret = prices.pct_change().fillna(0.0)
    vol_panel = _compute_rolling_vol(prices, window=lookback)

    equity = []
    dates = []
    # 从有足够历史开始走前测试
    for i in range(lookback, len(prices), rebalance_freq):
        d = prices.index[i]
        # 用截至 i-1 的信息打分/算vol，避免look-ahead（简单处理：用到i的截面也常演示用，这里用i-1更稳）
        score = score_fn(prices.iloc[:i]) if score_fn is not None else None
        if score is None or not isinstance(score, pd.Series) or score.empty:
            # 兜底：等权全市场
            sel = prices.columns
        else:
            cur_vol = vol_panel.iloc[i-1]  # 历史波动截面
            sel = select_top(score, top_n=top_n, vol_filter=vol_filter, vol=cur_vol)

        if len(sel) == 0:
            # 没选中就持现金（0收益）
            port_ret = 0.0
        else:
            # 等权（可用 max_weight 做上限，等权若超则缩放；这里演示用等权）
            w = pd.Series(1.0 / len(sel), index=sel)
            # 取当期月收益（i期相对i-1）
            month_ret = ret.iloc[i]
            port_ret = (w * month_ret.reindex(w.index).fillna(0.0)).sum()

        dates.append(d)
        equity.append(1.0 if len(equity) == 0 else equity[-1] * (1 + port_ret))

    equity = pd.Series(equity, index=pd.DatetimeIndex(dates), name="Equity")
    if len(equity) > 1:
        eq_ret = equity.pct_change().fillna(0.0)
        ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (12 / max(len(equity), 1)) - 1
        ann_vol = eq_ret.std(ddof=0) * np.sqrt(12)
        sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
        dd = equity / equity.cummax() - 1.0
        mdd = dd.min()
        stats = {
            "Total Return": equity.iloc[-1] / equity.iloc[0] - 1,
            "Annualized Return": ann_ret,
            "Annualized Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": mdd,
            "Periods": len(equity),
        }
    else:
        stats = {}

    return equity, stats