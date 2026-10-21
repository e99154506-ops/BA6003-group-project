# src/ai_insight.py
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _load_env() -> None:
    """Load environment variables from either the project root or src folder."""
    candidates = (
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    )
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)


_load_env()


def _client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )


def _normalize_yearly(yearly):
    """Accept both of these shapes:
    - {"2020": {"Strategy": 0.10, "SPY": 0.08}}
    - {"Strategy": {2020: 0.10}, "SPY": {2020: 0.08}}
    """
    if not isinstance(yearly, dict):
        return {}

    if "Strategy" in yearly and isinstance(yearly.get("Strategy"), dict):
        strategy_map = yearly.get("Strategy", {})
        spy_map = yearly.get("SPY", {})
        normalized = {}
        for year_key, strategy_val in strategy_map.items():
            year_label = str(year_key)
            normalized[year_label] = {
                "Strategy": float(strategy_val),
                "SPY": float(spy_map.get(year_key, 0.0)),
            }
        return normalized

    normalized = {}
    for year_key, value in yearly.items():
        if not isinstance(value, dict):
            continue
        strategy_val = value.get("Strategy", value.get("strategy", 0.0))
        spy_val = value.get("SPY", value.get("spy", 0.0))
        normalized[str(year_key)] = {
            "Strategy": float(strategy_val),
            "SPY": float(spy_val),
        }
    return normalized


def build_prompt(profile, start, end, stats, latest, yearly) -> str:
    """把当前回测结果拼成一段结构化 prompt"""
    stats_s = "\n".join(f"- {k}: {v:.2%}" if abs(v) < 100 else f"- {k}: {v:.2f}"
                       for k, v in stats.items() if isinstance(v, (int, float)))
    top = ", ".join(latest.get("tickers", [])[:10])
    yearly_map = _normalize_yearly(yearly)
    yr = "\n".join(
        f"- {k[:4]}: Strategy {v['Strategy']:.1%} / SPY {v['SPY']:.1%}"
        for k, v in list(yearly_map.items())[-5:]
    )

    return f"""You are an investment analyst preparing a professional investor memo based on a factor-based strategy backtest.

STRATEGY BACKTEST SUMMARY
- Risk profile: {profile}
- Period: {start} to {end}
- Universe: 50 mock tickers (monthly prices)
- Method: momentum + low-volatility value proxy, equal-weight top-N

PERFORMANCE STATS
{stats_s}

LATEST SELECTION (top scores): {top}

RECENT YEARLY RETURNS (Strategy vs SPY)
{yr}

Write a concise, polished investor memo in English (about 250-350 words) with direct investment implications. The output should read like a professional investment note for an informed investor.

Requirements:
1. Start with an overall assessment of the strategy’s attractiveness from an investor standpoint using total return, volatility, Sharpe, and drawdown.
2. Explain how the strategy behaved across different market environments and compare it directly with SPY.
3. Provide one credible rationale for the observed performance based on the factor design and market regime exposure.
4. Discuss the key risks and limitations of treating the backtest as investable evidence, including simulation assumptions and potential model overfitting.
5. End with a practical investment conclusion: whether the strategy appears compelling, risky, or premature as an investment idea.

Tone: crisp, analytical, professional, and credible. Do not claim this is investment advice. Present the results as research evidence from a historical backtest only."""


def generate_insight(profile, start, end, stats, latest, yearly,
                     model="deepseek/deepseek-chat", timeout=60) -> str:
    c = _client()
    prompt = build_prompt(profile, start, end, stats, latest, yearly)
    resp = c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()