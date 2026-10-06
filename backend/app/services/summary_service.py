"""
Generates a natural-language weekly summary for procurement managers.
Feeds the LLM exact, pre-computed numbers rather than asking it to
reason over raw data, so it cannot hallucinate figures that matter.

Supports two providers, picked via settings.llm_provider:
  - "gemini" (Google AI Studio -- free tier available)
  - "groq"   (Groq Cloud -- free tier, fast inference)

If no provider/key is configured, falls back to a deterministic
plain-text summary so the endpoint still works with zero setup.
"""

from app.core.config import settings


async def generate_weekly_summary(stats: dict) -> str:
    """
    stats is a structured dict assembled by the caller, e.g.:
    {
        "top_stores": [...], "bottom_stores": [...],
        "total_forecasted_demand": ..., "low_stock_alerts": [...],
        "week_start": "...", "week_end": "...",
    }
    """
    prompt = _build_prompt(stats)

    provider = settings.llm_provider.lower().strip()

    if provider == "gemini" and settings.gemini_api_key:
        return await _call_gemini(prompt)
    if provider == "groq" and settings.groq_api_key:
        return await _call_groq(prompt)

    return _fallback_summary(stats)


def _build_prompt(stats: dict) -> str:
    return (
        "Write a concise weekly summary for a retail procurement manager, "
        "using ONLY the numbers given below. Do not invent any figures.\n\n"
        f"Week: {stats.get('week_start')} to {stats.get('week_end')}\n"
        f"Top performing stores: {stats.get('top_stores')}\n"
        f"Lowest performing stores: {stats.get('bottom_stores')}\n"
        f"Total forecasted demand: {stats.get('total_forecasted_demand')}\n"
        f"Low stock alerts: {stats.get('low_stock_alerts')}\n\n"
        "Keep it to 3-4 short paragraphs, plain business language, no markdown headers."
    )


async def _call_gemini(prompt: str) -> str:
    """
    Uses Google's Gemini API (free tier at aistudio.google.com).
    Model: gemini-1.5-flash -- fast and free-tier friendly for this
    kind of short structured-text generation.
    """
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


async def _call_groq(prompt: str) -> str:
    """
    Uses Groq Cloud's OpenAI-compatible chat completions API
    (free tier at console.groq.com). Model: llama-3.1-8b-instant --
    cheap/fast, more than enough for this task.
    """
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


def _fallback_summary(stats: dict) -> str:
    return (
        f"Weekly summary ({stats.get('week_start')} to {stats.get('week_end')}): "
        f"Total forecasted demand across all stores is {stats.get('total_forecasted_demand')}. "
        f"{len(stats.get('low_stock_alerts', []))} stores flagged for low stock. "
        "(Set LLM_PROVIDER and the matching API key to enable full natural-language summaries.)"
    )
