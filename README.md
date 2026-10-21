# BA6003 Group Project

This repository contains a factor-based equity strategy dashboard and backtesting workflow for a university project. The project evaluates a monthly rebalanced portfolio that combines momentum and low-volatility signals, compares the strategy against SPY, and presents the results in an interactive Streamlit dashboard.

## Overview

The project is designed to demonstrate how a simple quantitative strategy can be:

- built from monthly price data,
- scored using factor signals,
- selected into a top-ranked basket,
- backtested against a benchmark,
- summarized in a professional investor-facing dashboard,
- and enhanced with AI-generated insight using DeepSeek via OpenRouter.

## Strategy Summary

The strategy uses a monthly factor approach:

- momentum signal based on 6-month price performance,
- low-volatility proxy as a value/quality-style filter,
- equal-weight allocation among the top selected names,
- benchmark comparison against SPY,
- risk and performance metrics such as return, volatility, Sharpe ratio, and drawdown.

The dashboard allows users to adjust the strategy profile (Conservative / Balanced / Aggressive) and inspect the resulting portfolio composition and performance.

## Project Structure

```text
BA6003-group-project/
├── data/
│   └── raw/
│       ├── prices_monthly.csv
│       └── spy_monthly.csv
├── src/
│   ├── ai_insight.py
│   ├── backtest.py
│   ├── config.py
│   ├── dashboard.py
│   ├── data_loader.py
│   ├── factor_model.py
│   ├── generate_mock_data.py
│   ├── main.py
│   └── metrics.py
├── tests/
│   ├── test_ai_insight.py
│   ├── test_backtest.py
│   ├── test_data_loader.py
│   └── test_factor_model.py
├── outputs/
├── README.md
├── .gitignore
└── .env   # local environment file, not committed
```

## Setup

Create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy matplotlib python-dotenv openai pytest
```

Create a local `.env` file at the project root and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
```

The AI insight feature uses the model `deepseek/deepseek-chat` through OpenRouter.

## Run the Dashboard

From the project root:

```bash
source .venv/bin/activate
streamlit run src/dashboard.py
```

The dashboard provides:

- strategy profile selection,
- portfolio performance charts,
- benchmark comparison,
- top ticker selection summary,
- AI-generated research memo.

## Run the Backtest / Model Logic

The project can also be executed directly from code:

```bash
source .venv/bin/activate
PYTHONPATH=. python src/main.py
```

## Run Tests

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q
```

The test suite verifies the data loader, factor calculations, portfolio selection logic, and backtest behavior.

## Notes for GitHub Review

This repository is organised for review by instructors and evaluators. The code is structured around data preparation, factor scoring, portfolio construction, performance evaluation, and dashboard presentation, so that the full pipeline can be inspected in one place without needing hidden setup steps.

## Security Reminder

Do not commit `.env` files or API keys to GitHub. The project relies on local environment variables for AI functionality.
