import pandas as pd
import matplotlib.pyplot as plt
from .data_loader import load_prices, compute_monthly_returns
from .factor_model import build_price_factors, build_ff3_factors, ff3_regression, select_top_portfolio
from .backtest import build_weights_from_selection, run_backtest
from .metrics import performance_summary
from .config import FACTOR_METHOD

def main():
    prices = load_prices()
    monthly_ret, monthly_close = compute_monthly_returns(prices)
    
    price_factors = build_price_factors(monthly_close)
    ff3 = build_ff3_factors(monthly_ret.index)
    regression_results = ff3_regression(monthly_ret, ff3)
    
    latest_date = price_factors["date"].max()
    latest_factors = price_factors[price_factors["date"] == latest_date]
    combined = regression_results.merge(latest_factors, on="symbol", how="left")
    
    top_portfolio = select_top_portfolio(combined, method=FACTOR_METHOD)
    print(f"\n📊 选中 {len(top_portfolio)} 只股票:")
    print(top_portfolio[["symbol", "score", "weight"]].head(10).to_string(index=False))
    
    weights = build_weights_from_selection(top_portfolio, monthly_ret)
    portfolio_ret = run_backtest(weights, monthly_ret)
    
    stats = performance_summary(portfolio_ret)
    print(f"\n📈 回测结果 ({FACTOR_METHOD} 模式)")
    for k, v in stats.items():
        print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
    
    cum_return = (1 + portfolio_ret).cumprod()
    plt.figure(figsize=(10, 5))
    cum_return.plot(title=f"策略累计净值 ({FACTOR_METHOD})")
    plt.ylabel("Net Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()