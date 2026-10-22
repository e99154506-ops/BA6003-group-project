import os

from src import ai_insight


def test_load_env_reads_project_dotenv(monkeypatch):
    project_env = ai_insight.Path(ai_insight.__file__).resolve().parent.parent / ".env"
    loaded_paths = []

    monkeypatch.setattr(ai_insight.Path, "exists", lambda path: path == project_env)
    monkeypatch.setattr(ai_insight, "load_dotenv", lambda path: loaded_paths.append(path))
    ai_insight._load_env()

    assert loaded_paths == [project_env]


def test_build_prompt_handles_dataframe_yearly_dict_shape():
    yearly = {
        "Strategy": {2020: 0.12, 2021: 0.08},
        "SPY": {2020: 0.07, 2021: 0.05},
    }

    prompt = ai_insight.build_prompt(
        profile="Balanced",
        start="2020-01-01",
        end="2021-12-31",
        stats={"Return": 0.1, "Volatility": 0.2},
        latest={"tickers": ["AAPL", "MSFT"]},
        yearly=yearly,
    )

    assert "2020" in prompt
    assert "Strategy" in prompt
    assert "SPY" in prompt
