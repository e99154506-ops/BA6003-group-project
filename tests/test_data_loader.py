import pandas as pd
import pytest

from src import data_loader


def test_load_prices_renames_columns_sorts_rows_and_converts_close(
    tmp_path, monkeypatch
):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Symbol": ["B", "A"],
            "Date": ["2023-02-28", "2023-01-31"],
            "Close": ["20", "10"],
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(data_loader, "PRICE_CSV", csv_path)

    result = data_loader.load_prices()

    assert list(result.columns) == ["symbol", "date", "close"]
    assert result["symbol"].tolist() == ["A", "B"]
    assert pd.api.types.is_numeric_dtype(result["close"])


def test_load_prices_raises_when_file_is_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.csv"
    monkeypatch.setattr(data_loader, "PRICE_CSV", missing_path)

    with pytest.raises(FileNotFoundError):
        data_loader.load_prices()


def test_compute_monthly_returns_calculates_returns_from_month_end_prices():
    dates = pd.to_datetime(
        ["2023-01-31", "2023-02-28", "2023-03-31"] * 2
    )
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["A"] * 3 + ["B"] * 3,
            "close": [100, 110, 99, 200, 180, 198],
        }
    )

    monthly_returns, monthly_close = data_loader.compute_monthly_returns(prices)

    assert monthly_close.loc[pd.Timestamp("2023-01-31"), "A"] == 100
    assert monthly_returns.loc[pd.Timestamp("2023-02-28"), "A"] == pytest.approx(0.10)
    assert monthly_returns.loc[pd.Timestamp("2023-03-31"), "A"] == pytest.approx(-0.10)
    assert monthly_returns.loc[pd.Timestamp("2023-02-28"), "B"] == pytest.approx(-0.10)
    assert monthly_returns.loc[pd.Timestamp("2023-03-31"), "B"] == pytest.approx(0.10)