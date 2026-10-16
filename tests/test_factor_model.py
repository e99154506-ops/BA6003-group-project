import numpy as np
import pandas as pd
import pytest

from src import factor_model


def test_build_price_factors_calculates_six_month_momentum():
    dates = pd.date_range("2023-01-31", periods=8, freq="ME")
    monthly_close = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 100, 100, 100, 200, 220],
            "BBB": [100, 110, 120, 130, 140, 150, 160, 170],
        },
        index=dates,
    )

    factors = factor_model.build_price_factors(monthly_close)

    aaa_at_six_months = factors[
        (factors["symbol"] == "AAA")
        & (factors["date"] == dates[6])
    ]
    assert len(aaa_at_six_months) == 1
    assert aaa_at_six_months.iloc[0]["momentum_6m"] == pytest.approx(1.0)
    assert {"date", "symbol", "momentum_6m", "volatility_6m"} <= set(factors.columns)


def test_build_ff3_factors_is_reproducible():
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")

    first = factor_model.build_ff3_factors(dates)
    second = factor_model.build_ff3_factors(dates)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert {"date", "Mkt-RF", "SMB", "HML", "RF"} <= set(first.columns)


def test_ff3_regression_recovers_known_coefficients():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")

    market = rng.normal(0.01, 0.04, len(dates))
    smb = rng.normal(0.00, 0.02, len(dates))
    hml = rng.normal(0.00, 0.02, len(dates))
    risk_free = np.full(len(dates), 0.001)

    alpha = 0.002
    beta_market = 1.2
    beta_smb = -0.4
    beta_hml = 0.7

    excess_returns = (
        alpha
        + beta_market * market
        + beta_smb * smb
        + beta_hml * hml
    )

    monthly_returns = pd.DataFrame(
        {"AAA": excess_returns + risk_free},
        index=dates,
    )
    ff3 = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": market,
            "SMB": smb,
            "HML": hml,
            "RF": risk_free,
        }
    )

    result = factor_model.ff3_regression(monthly_returns, ff3)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["alpha"] == pytest.approx(alpha * 12)
    assert row["beta_mkt"] == pytest.approx(beta_market)
    assert row["beta_smb"] == pytest.approx(beta_smb)
    assert row["beta_hml"] == pytest.approx(beta_hml)


def test_ff3_regression_skips_symbols_with_fewer_than_ten_observations():
    dates = pd.date_range("2023-01-31", periods=9, freq="ME")
    monthly_returns = pd.DataFrame(
        {"AAA": np.linspace(0.01, 0.09, len(dates))},
        index=dates,
    )
    ff3 = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": np.linspace(0.01, 0.09, len(dates)),
            "SMB": np.linspace(0.02, 0.10, len(dates)),
            "HML": np.linspace(0.03, 0.11, len(dates)),
            "RF": 0.001,
        }
    )

    result = factor_model.ff3_regression(monthly_returns, ff3)

    assert result.empty


def test_select_top_portfolio_alpha_method(monkeypatch):
    monkeypatch.setattr(factor_model, "TOP_N", 2)

    factors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "alpha": [0.10, 0.20, 0.05],
            "beta_mkt": [1.0, 1.0, 1.0],
        }
    )

    result = factor_model.select_top_portfolio(factors, method="alpha")

    assert result["symbol"].tolist() == ["BBB", "AAA"]
    assert result["weight"].sum() == pytest.approx(1.0)


def test_select_top_portfolio_momentum_method(monkeypatch):
    monkeypatch.setattr(factor_model, "TOP_QUANTILE", 0.5)

    factors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "momentum_6m": [0.10, 0.40, 0.20, 0.30],
        }
    )

    result = factor_model.select_top_portfolio(factors, method="momentum")

    assert result["symbol"].tolist() == ["BBB", "DDD"]
    assert result["weight"].sum() == pytest.approx(1.0)


def test_select_top_portfolio_combo_method(monkeypatch):
    monkeypatch.setattr(factor_model, "TOP_QUANTILE", 0.5)

    factors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "alpha": [0.10, 0.20, 0.05, 0.15],
            "momentum_6m": [0.20, 0.30, 0.10, 0.25],
            "volatility_6m": [0.10, 0.10, 0.20, 0.10],
        }
    )

    result = factor_model.select_top_portfolio(factors, method="combo")

    assert len(result) == 2
    assert result["score"].is_monotonic_decreasing
    assert result["weight"].sum() == pytest.approx(1.0)


def test_select_top_portfolio_rejects_unknown_method():
    factors = pd.DataFrame({"symbol": ["AAA"]})

    with pytest.raises(ValueError):
        factor_model.select_top_portfolio(factors, method="unknown")