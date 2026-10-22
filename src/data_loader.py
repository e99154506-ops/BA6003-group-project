from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PRICE_CSV = DATA_DIR / "prices_monthly.csv"


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw price data in the legacy contract used by the project tests."""
    csv_path = Path(path) if path is not None else PRICE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file missing: {csv_path}. "
            "Run: `python -m src.generate_mock_data` from project root."
        )

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        else:
            raise KeyError("Expected a 'symbol' column in the price file.")
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            raise KeyError("Expected a 'date' column in the price file.")
    if "close" not in df.columns:
        for candidate in ("price", "adj close", "adj_close"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "close"})
                break
        else:
            raise KeyError("Expected a 'close' column in the price file.")

    df = df[["symbol", "date", "close"]].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file missing: {path}. "
            "Run: `python -m src.generate_mock_data` from project root."
        )
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def load_monthly() -> pd.DataFrame:
    return _read(DATA_DIR / "prices_monthly.csv")


def load_spy_benchmark() -> pd.Series:
    return _read(DATA_DIR / "spy_monthly.csv")["SPY"]


def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    return df.loc[(df.index >= start_date) & (df.index <= end_date)]


def compute_monthly_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"symbol", "date", "close"}.issubset(prices.columns):
        raise KeyError("Expected columns: symbol, date, close")

    clean = prices[["symbol", "date", "close"]].copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["close"]).sort_values(["date", "symbol"])

    monthly_close = clean.pivot(index="date", columns="symbol", values="close").sort_index()
    monthly_returns = monthly_close.pct_change().fillna(0.0)
    return monthly_returns, monthly_close
