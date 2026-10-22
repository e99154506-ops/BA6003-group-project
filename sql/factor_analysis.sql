CREATE OR REPLACE VIEW stock_prices_long AS
SELECT
    date::DATE AS date,
    symbol,
    close::DOUBLE AS close
FROM read_csv_auto('data/raw/prices_monthly.csv', header = true)
UNPIVOT (
    close FOR symbol IN (COLUMNS(* EXCLUDE (date)))
);

-- Calculate each ticker's monthly return and its next-month return.
CREATE OR REPLACE VIEW stock_monthly_returns AS
WITH returns AS (
    SELECT
        date,
        symbol,
        close,
        close / NULLIF(LAG(close) OVER ticker_months, 0) - 1 AS monthly_return
    FROM stock_prices_long
    WINDOW ticker_months AS (PARTITION BY symbol ORDER BY date)
)
SELECT
    date,
    symbol,
    close,
    monthly_return,
    LEAD(date) OVER ticker_months AS next_date,
    LEAD(monthly_return) OVER ticker_months AS next_month_return
FROM returns
WINDOW ticker_months AS (PARTITION BY symbol ORDER BY date);

-- Build six-month momentum and six-observation realized volatility signals.
CREATE OR REPLACE VIEW stock_factor_signals AS
WITH price_history AS (
    SELECT
        date,
        symbol,
        close,
        monthly_return,
        next_date,
        next_month_return,
        close / NULLIF(LAG(close, 6) OVER ticker_months, 0) - 1 AS momentum_6m,
        STDDEV_SAMP(monthly_return) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS volatility_6m
    FROM stock_monthly_returns
    WINDOW ticker_months AS (PARTITION BY symbol ORDER BY date)
)
SELECT *
FROM price_history
WHERE momentum_6m IS NOT NULL
  AND volatility_6m IS NOT NULL;

-- Rank momentum and low volatility within each month, then combine the ranks.
CREATE OR REPLACE VIEW ranked_stocks AS
WITH factor_ranks AS (
    SELECT
        *,
        PERCENT_RANK() OVER (
            PARTITION BY date ORDER BY momentum_6m
        ) AS momentum_rank,
        PERCENT_RANK() OVER (
            PARTITION BY date ORDER BY volatility_6m DESC
        ) AS low_volatility_rank
    FROM stock_factor_signals
)
SELECT
    *,
    (momentum_rank + low_volatility_rank) / 2 AS combined_score,
    ROW_NUMBER() OVER (
        PARTITION BY date
        ORDER BY (momentum_rank + low_volatility_rank) DESC, symbol
    ) AS selection_rank
FROM factor_ranks;

-- Select the 15 highest-ranked tickers each month and apply equal weights.
CREATE OR REPLACE VIEW strategy_monthly_returns AS
SELECT
    next_date AS date,
    AVG(next_month_return) AS strategy_return,
    COUNT(*) AS holdings_count
FROM ranked_stocks
WHERE selection_rank <= 15
  AND next_date IS NOT NULL
  AND next_month_return IS NOT NULL
GROUP BY next_date;

-- Calculate monthly SPY returns and compare them with strategy returns.
CREATE OR REPLACE VIEW benchmark_monthly_returns AS
WITH spy_prices AS (
    SELECT date::DATE AS date, SPY::DOUBLE AS close
    FROM read_csv_auto('data/raw/spy_monthly.csv', header = true)
),
spy_returns AS (
    SELECT
        date,
        close / NULLIF(LAG(close) OVER (ORDER BY date), 0) - 1 AS spy_return
    FROM spy_prices
)
SELECT date, spy_return
FROM spy_returns
WHERE spy_return IS NOT NULL;

-- Final result: compounded calendar-year returns and the number of months
-- included in each year's comparison.
WITH monthly_comparison AS (
    SELECT
        YEAR(strategy.date) AS year,
        strategy.strategy_return,
        benchmark.spy_return,
        strategy.holdings_count
    FROM strategy_monthly_returns AS strategy
    INNER JOIN benchmark_monthly_returns AS benchmark USING (date)
)
SELECT
    year,
    EXP(SUM(LN(GREATEST(1 + strategy_return, 1e-12)))) - 1
        AS strategy_return,
    EXP(SUM(LN(GREATEST(1 + spy_return, 1e-12)))) - 1
        AS spy_return,
    EXP(SUM(LN(GREATEST(1 + strategy_return, 1e-12))))
      - EXP(SUM(LN(GREATEST(1 + spy_return, 1e-12))))
        AS excess_return,
    COUNT(*) AS months,
    ROUND(AVG(holdings_count), 1) AS average_holdings
FROM monthly_comparison
GROUP BY year
ORDER BY year;