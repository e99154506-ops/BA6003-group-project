# src/config.py  （顶层）

RISK_PROFILES = {
    "Conservative": {
        "top_n": 10,
        "vol_filter": 0.35,
        "max_weight": 0.08,
        "method": "combo",
        "mom_weight": 0.4,
        "val_weight": 0.6,
    },
    "Balanced": {
        "top_n": 15,
        "vol_filter": 0.45,
        "max_weight": 0.06,
        "method": "combo",
        "mom_weight": 0.6,
        "val_weight": 0.4,
    },
    "Aggressive": {
        "top_n": 20,
        "vol_filter": 0.60,
        "max_weight": 0.05,
        "method": "momentum",
        "mom_weight": 0.8,
        "val_weight": 0.2,
    },
}

# 给回测/数据筛选用的日期字符串
START_DEFAULT = "2015-01-01"
END_DEFAULT   = "2024-12-31"

# 给侧边栏“年份选择”用的整数（避免再对 range(2015,2025).index("2015-01-01")）
START_YEAR_DEFAULT = 2015
END_YEAR_DEFAULT   = 2024

# Available universe / data assumptions for mock generator
TICKERS = [f"T{i:03d}" for i in range(1, 51)]  # T001..T050

# SPY file path helper (relative to project root)
SPY_PATH = "data/raw/spy_monthly.csv"