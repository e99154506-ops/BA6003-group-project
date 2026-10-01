# src/main.py
import logging
from .data_loader import download_prices, compute_monthly_returns, save_outputs
from .factor_model import build_ff3_factors, ff3_regression, select_top_portfolio
from .backtest import run_backtest
from .config import OUTPUTS_DIR

logging.basicConfig(
    filename=OUTPUTS_DIR / "pipeline.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Pipeline started")
    
    # 1. 数据
    prices = download_prices()
    monthly = compute_monthly_returns(prices)
    save_outputs(prices, monthly)
    logger.info(f"Data saved: {len(prices)} rows")
    
    # 2. 因子+选股
    ff3 = build_ff3_factors(monthly["date"].unique())
    factor_df = ff3_regression(monthly, ff3)
    top = select_top_portfolio(factor_df)
    top.to_csv(OUTPUTS_DIR / "sample_demo_portfolio.csv", index=False)
    logger.info(f"Top {len(top)} selected")
    
    # 3. 回测
    run_backtest(monthly, top)
    logger.info("Backtest complete")
    
    print("✅ Pipeline finished. Check outputs/ and pipeline.log")

if __name__ == "__main__":
    main()