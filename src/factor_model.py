# src/factor_model.py
import numpy as np
import pandas as pd

TOP_N = 10
TOP_QUANTILE = 0.2


def build_price_factors(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """Create a long-form factor table containing momentum and volatility signals."""
    if monthly_close.empty:
        return pd.DataFrame(columns=["date", "symbol", "momentum_6m", "volatility_6m"])

    frames: list[dict] = []
    for symbol in monthly_close.columns:
        series = pd.to_numeric(monthly_close[symbol], errors="coerce")
        momentum = (series / series.shift(6) - 1.0).fillna(0.0)
        volatility = series.pct_change().rolling(6, min_periods=1).std().fillna(0.0)
        for date, mom_val in momentum.items():
            frames.append({
                "date": pd.Timestamp(date),
                "symbol": str(symbol),
                "momentum_6m": float(mom_val),
                "volatility_6m": float(volatility.loc[date]),
            })

    return pd.DataFrame(frames).sort_values(["date", "symbol"]).reset_index(drop=True)


def build_ff3_factors(dates) -> pd.DataFrame:
    """Return a deterministic Fama-French 3-factor matrix for the supplied dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    rng = np.random.default_rng(0)
    market = rng.normal(0.01, 0.04, len(idx))
    smb = rng.normal(0.00, 0.02, len(idx))
    hml = rng.normal(0.00, 0.02, len(idx))
    risk_free = np.full(len(idx), 0.001)
    return pd.DataFrame({
        "date": idx,
        "Mkt-RF": market,
        "SMB": smb,
        "HML": hml,
        "RF": risk_free,
    })


def ff3_regression(monthly_returns: pd.DataFrame, ff3: pd.DataFrame) -> pd.DataFrame:
    """Run a cross-sectional OLS regression of each symbol's excess return on FF3 factors."""
    if monthly_returns.empty or ff3.empty:
        return pd.DataFrame(columns=["symbol", "alpha", "beta_mkt", "beta_smb", "beta_hml"])
    if len(monthly_returns) < 10:
        return pd.DataFrame(columns=["symbol", "alpha", "beta_mkt", "beta_smb", "beta_hml"])

    ff3 = ff3.copy().sort_values("date").reset_index(drop=True)
    ff3_index = ff3.set_index("date")
    aligned = monthly_returns.reindex(ff3_index.index)
    if aligned.empty:
        return pd.DataFrame(columns=["symbol", "alpha", "beta_mkt", "beta_smb", "beta_hml"])

    design = np.column_stack([
        np.ones(len(aligned)),
        ff3_index["Mkt-RF"].to_numpy(dtype=float),
        ff3_index["SMB"].to_numpy(dtype=float),
        ff3_index["HML"].to_numpy(dtype=float),
    ])

    rows = []
    for symbol in aligned.columns:
        y = aligned[symbol].to_numpy(dtype=float) - ff3_index["RF"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rows.append({
            "symbol": str(symbol),
            "alpha": float(coef[0] * 12.0),
            "beta_mkt": float(coef[1]),
            "beta_smb": float(coef[2]),
            "beta_hml": float(coef[3]),
        })

    return pd.DataFrame(rows)


def _score_portfolio(factors: pd.DataFrame, method: str) -> pd.DataFrame:
    if "symbol" not in factors.columns:
        raise ValueError("Factor table must contain a 'symbol' column.")

    if method == "alpha":
        if "alpha" not in factors.columns:
            raise ValueError("Alpha method requires an 'alpha' column.")
        score_col = "alpha"
    elif method == "momentum":
        if "momentum_6m" not in factors.columns:
            raise ValueError("Momentum method requires a 'momentum_6m' column.")
        score_col = "momentum_6m"
    elif method == "combo":
        if {"alpha", "momentum_6m", "volatility_6m"}.difference(factors.columns):
            raise ValueError("Combo method requires 'alpha', 'momentum_6m', and 'volatility_6m'.")
        score_col = "combo_score"
        factors = factors.copy()
        factors[score_col] = factors["alpha"] + factors["momentum_6m"] - factors["volatility_6m"]
    else:
        raise ValueError(f"Unknown method: {method}")

    ranking = factors[["symbol", score_col]].dropna().copy()
    ranking = ranking.sort_values(score_col, ascending=False).reset_index(drop=True)
    if method == "momentum":
        k = max(1, int(np.ceil(len(ranking) * TOP_QUANTILE)))
        ranking = ranking.head(k)
    elif method == "alpha":
        k = min(len(ranking), TOP_N)
        ranking = ranking.head(k)
    elif method == "combo":
        k = max(1, min(len(ranking), int(np.ceil(len(ranking) * TOP_QUANTILE))))
        ranking = ranking.head(k)

    ranking = ranking.reset_index(drop=True)
    n = len(ranking)
    if n == 0:
        return pd.DataFrame(columns=["symbol", "score", "weight"])

    ranking["score"] = ranking[score_col].astype(float)
    ranking["weight"] = 1.0 / n
    return ranking[["symbol", "score", "weight"]]


def select_top_portfolio(factors: pd.DataFrame, method: str = "combo") -> pd.DataFrame:
    """Select top-ranked symbols and assign equal weights to the selected basket."""
    return _score_portfolio(factors.copy(), method)

def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """对每一期（行）做截面 z-score：(x - 行均值) / 行标准差，防除零"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)

def _momentum_signal(prices: pd.DataFrame, lookback=12, skip=1) -> pd.DataFrame:
    """
    月频动量：用 t-lookback 到 t-skip 的累计收益代理（避免最近月反转污染）
    返回每期截面动量 z-score
    """
    # 用价格算区间收益：P_{t-skip}/P_{t-lookback} - 1
    ret = prices.shift(skip) / prices.shift(lookback) - 1
    return _cross_sectional_zscore(ret)

def _value_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """
    纯价量下没有 PE/PB，用“低波动/回撤浅”当价值/质量代理（越高越优）
    这里用 -滚动波动率（截面z）作为代理，仅用于演示框架
    """
    vol = prices.pct_change().rolling(12).std()
    return _cross_sectional_zscore(-vol)

def score_factors(
    prices_df: pd.DataFrame,
    method: str = "combo",
    mom_weight: float = 0.6,
    val_weight: float = 0.4,
) -> pd.Series:
    """
    输入：月频价格面板 (index=date, columns=tickers)
    输出：最近一期的截面综合得分 Series（index=tickers）
    """
    if prices_df.empty or prices_df.shape[1] == 0:
        return pd.Series(dtype=float)

    # 至少需要足够历史
    if len(prices_df) < 13:
        return pd.Series(0.0, index=prices_df.columns)

    mom_z = _momentum_signal(prices_df, lookback=12, skip=1)
    val_z = _value_proxy(prices_df)

    if method == "momentum":
        score = mom_z
    elif method == "value":
        score = val_z
    else:  # combo
        # 权重归一化
        s = (mom_weight + val_weight) or 1.0
        score = mom_z * (mom_weight / s) + val_z * (val_weight / s)

    # 取最后一期截面得分
    last = score.iloc[-1].dropna()
    # 再做一次横截面排名归一化到大致 [-1,1]（更稳）
    if not last.empty and last.std(ddof=0) and last.std(ddof=0) > 0:
        last = (last - last.mean()) / last.std(ddof=0)
    return last