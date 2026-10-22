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


def test_client_uses_streamlit_cloud_secret_when_environment_key_is_missing(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ai_insight, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        ai_insight,
        "_load_env",
        lambda: None,
    )

    import streamlit as st

    monkeypatch.setattr(st, "secrets", {"OPENROUTER_API_KEY": "cloud-test-key"})

    ai_insight._client()

    assert captured["api_key"] == "cloud-test-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
