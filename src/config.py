from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据下载
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'NVDA']
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
OUTPUTS_DIR = DATA_DIR
PRICE_CSV = DATA_DIR / "sp500_prices.csv"

# 回测
LOOKBACK = 6
TOP_QUANTILE = 0.1
TOP_N = 1
RISK_FREE_RATE = 0.02
FACTOR_METHOD = "combo"

# 中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False