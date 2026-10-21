import pandas as pd
import numpy as np

def perf_stats(returns: pd.Series, bench: pd.Series = None, periods_per_year=12):
    r = returns.dropna()
    if len(r) == 0:
        return {}
    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    ann = cum.iloc[-1] ** (periods_per_year / len(r)) - 1
    vol = r.std() * (periods_per_year ** 0.5)
    sharpe = ann / vol if vol > 0 else 0
    dd = (cum / cum.cummax() - 1).min()

    out = {
        "Total Return": total,
        "Annualized Return": ann,
        "Annualized Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": dd,
        "N Months": len(r),
    }

    if bench is not None:
        b = bench.reindex(r.index).dropna()
        if len(b) > 0:
            ar = r - b
            info = ar.mean() / ar.std() * (periods_per_year ** 0.5) if ar.std()>0 else 0
            beat = (r > b).mean()
            out["Excess Total"] = cum.iloc[-1] - (1+b).cumprod().iloc[-1]
            out["Information Ratio"] = info
            out["Hit Rate vs SPY"] = beat

    return out

def fmt(v):
    if pd.isna(v): return "—"
    if isinstance(v, float):
        return f"{v:.2%}" if abs(v)<10 else f"{v:.2f}"
    return str(v)