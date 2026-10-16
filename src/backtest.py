import pandas as pd
import numpy as np

def build_weights_from_selection(top_portfolio: pd.DataFrame, monthly_ret: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=monthly_ret.index, columns=monthly_ret.columns)
    selected = top_portfolio["symbol"].tolist()
    w = 1.0 / len(selected)
    
    for date in weights.index[6:]:
        for sym in selected:
            if sym in weights.columns:
                weights.loc[date, sym] = w
    return weights

def run_backtest(weights: pd.DataFrame, monthly_ret: pd.DataFrame) -> pd.Series:
    return (weights.shift(1) * monthly_ret).sum(axis=1).dropna()