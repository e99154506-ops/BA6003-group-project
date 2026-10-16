import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.backtest import build_weights_from_selection, run_backtest


def test_run_backtest_uses_previous_month_weights():
    dates = pd.date_range("2023-01-31", periods=3, freq="ME")

    weights = pd.DataFrame(
        {
            "A": [1.0, 0.5, 0.0],
            "B": [0.0, 0.5, 1.0],
        },
        index=dates,
    )
    monthly_returns = pd.DataFrame(
        {
            "A": [0.10, 0.20, 0.30],
            "B": [0.20, 0.10, -0.10],
        },
        index=dates,
    )

    result = run_backtest(weights, monthly_returns)
    expected = pd.Series(
        [0.0, 0.20, 0.10],
        index=dates,
        dtype=float,
    )

    assert_series_equal(result, expected)


def test_build_weights_starts_after_six_months_and_is_equal_weighted():
    dates = pd.date_range("2023-01-31", periods=8, freq="ME")
    monthly_returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=["A", "B", "C"],
    )
    selected_stocks = pd.DataFrame({"symbol": ["A", "B"]})

    weights = build_weights_from_selection(selected_stocks, monthly_returns)

    assert weights.loc[dates[:6]].to_numpy().sum() == 0
    np.testing.assert_allclose(weights.loc[dates[6], ["A", "B"]], [0.5, 0.5])
    assert weights.loc[dates[6], "C"] == 0
    np.testing.assert_allclose(weights.loc[dates[7], ["A", "B"]], [0.5, 0.5])


def test_backtest_returns_zero_when_previous_weights_are_zero():
    dates = pd.date_range("2023-01-31", periods=2, freq="ME")
    weights = pd.DataFrame({"A": [0.0, 1.0]}, index=dates)
    monthly_returns = pd.DataFrame({"A": [0.10, 0.20]}, index=dates)

    result = run_backtest(weights, monthly_returns)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.0