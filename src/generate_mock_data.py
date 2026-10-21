# src/generate_mock_data.py
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICES_PATH = DATA_DIR / "prices_monthly.csv"
SPY_PATH = DATA_DIR / "spy_monthly.csv"


def main():
    # 用 ME（month-end），不要用 M（pandas 2.2+/3.x 会报 Invalid frequency: 'M'）
    dates = pd.date_range("2015-01-31", periods=120, freq="ME")
    tickers = [f"T{i:03d}" for i in range(50)]

    rng = np.random.default_rng(42)
    p = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for c in tickers:
        ret = rng.normal(0.008, 0.06, size=len(dates))
        p[c] = 100 * (1 + ret).cumprod()

    p.to_csv(PRICES_PATH, index_label="date")

    spy_ret = rng.normal(0.007, 0.045, size=len(dates))
    spy = pd.Series(100 * (1 + spy_ret).cumprod(), index=dates, name="SPY")
    spy.to_frame().to_csv(SPY_PATH, index_label="date")

    print(f"Wrote {PRICES_PATH}")
    print(f"Wrote {SPY_PATH}")


if __name__ == "__main__":
    main()