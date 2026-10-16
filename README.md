# BA6003 Group Project

## Project Structure

- `src/` — Data loading, factor model, backtest, configuration, and dashboard code
- `data/raw/` — Raw price data used by the project
- `outputs/` — Generated charts and results
- `tests/` — Unit tests for data loading, factor calculations, portfolio selection, and backtesting

## Streamlit Dashboard

The dashboard reads raw prices from `data/raw/sp500_prices.csv` and calculates monthly returns through the data loader.

Run from the project root:

```bash
.venv/bin/python -m streamlit run src/dashboard.py
```

## Run Tests

```bash
.venv/bin/python -m pytest tests -v
```

The current test suite covers data loading, monthly returns, factor calculations, portfolio selection, and backtest weight timing.

## API Key Security

Store API keys in a local `.env` file. Do not commit `.env` files or expose API keys in source code.
