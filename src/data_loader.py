import pandas as pd
from pathlib import Path
from .config import PRICE_CSV

def load_prices() -> pd.DataFrame:
    """Read the local price CSV."""
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_CSV}")

    prices = pd.read_csv(PRICE_CSV)
    prices = prices.rename(
        columns={"Symbol": "symbol", "Date": "date", "Close": "close"}
    )

    required = {"symbol", "date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price CSV is missing columns: {sorted(missing)}")

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="raise")
    return prices

def compute_monthly_returns(prices: pd.DataFrame):
    """Returns monthly returns wide table + end-of-month closing prices wide table."""
    monthly_close = (
        prices.set_index("date")
        .groupby("symbol")["close"]
        .resample("ME").last()
        .unstack("symbol")
    )
    monthly_ret = monthly_close.pct_change()
    return monthly_ret, monthly_close