# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 确保输出目录存在
OUTPUTS_DIR.mkdir(exist_ok=True)

# 股票参数
TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA"]
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"

# FF3 模拟参数
RISK_FREE_RATE = 0.0003
TOP_N = 3