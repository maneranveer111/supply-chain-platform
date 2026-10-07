"""
LLM Service & Provider Abstraction for Supply Chain Business Insights.

Architectural invariants:
  - Clean provider interface supporting Gemini (primary) and Groq (optional).
  - Used strictly for natural-language business insights, explanations, and summaries.
  - Numerical forecasting remains the exclusive responsibility of the ML models.
  - Any figures in prompts come strictly from verified database/forecast state.
  - Failures, timeouts, and rate limits degrade gracefully to deterministic fallback text.
  - Secrets and API keys are never logged or exposed in responses.
"""

import logging
from typing import Any, Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        ...


class GeminiProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.model_name = model_name

    async def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        # Run generation asynchronously via thread pool if needed, or call generate_content
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.2},
        )
        if response and response.text:
            return response.text.strip()
        raise ValueError("Empty response received from Gemini API.")


class GroqProvider:
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model_name = model_name

    async def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        raise ValueError("Empty response received from Groq API.")


class FallbackProvider:
    async def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        return (
            "System Notice: Automated natural language generation is currently using "
            "deterministic business rules. Configure LLM_PROVIDER and matching credentials "
            "for dynamic generative insights."
        )

    async def generate_insight(self, prompt: str, fallback_text: str = "", max_tokens: int = 600) -> str:
        return fallback_text or await self.generate_text(prompt, max_tokens=max_tokens)


def get_llm_provider() -> LLMProvider:
    """
    Factory to resolve the active LLM provider based on application settings.
    """
    provider_name = (settings.llm_provider or "").lower().strip()

    if provider_name == "gemini" and settings.gemini_api_key:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model or "gemini-3.8-flash",
        )
    elif provider_name == "groq" and settings.groq_api_key:
        return GroqProvider(api_key=settings.groq_api_key)
    else:
        return FallbackProvider()


async def safe_generate_insight(prompt: str, fallback_text: str, max_tokens: int = 600) -> str:
    """
    Executes an LLM generation call with complete error handling.
    If the provider encounters any exception (network, quota, timeout),
    logs the event without leaking credentials and returns fallback_text.
    """
    provider = get_llm_provider()
    if isinstance(provider, FallbackProvider):
        return fallback_text

    try:
        return await provider.generate_text(prompt, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning(
            "LLM generation failed via provider '%s': %s. Falling back to deterministic summary.",
            settings.llm_provider,
            exc,
        )
        return fallback_text


# ---------------------------------------------------------------------------
# Business Insight Domain Functions
# ---------------------------------------------------------------------------

async def explain_forecast(store_name_or_id: Any = None, forecast_data: dict | None = None, **kwargs) -> str:
    """
    Explains a store's sales forecast in business language based strictly
    on verified predictions. Accepts either (store_name, forecast_dict)
    or keyword arguments.
    """
    store_name = str(store_name_or_id or kwargs.get("store_id") or "Store")
    if not store_name.startswith("Store"):
        store_name = f"Store #{store_name}"

    data = forecast_data or kwargs
    predictions = data.get("forecast", [])
    method = data.get("forecast_method", "unknown")
    confidence = data.get("forecast_confidence", "standard")
    horizon = data.get("horizon") or data.get("horizon_days") or (len(predictions) if predictions else 7)

    if method == "insufficient_data":
        return (
            f"Forecast explanation for {store_name}: Insufficient historical data is currently available "
            f"to generate statistical sales predictions. Onboarding data ingestion or benchmark assignment "
            f"is required before forecasting can commence."
        )

    if predictions:
        total_pred = sum(p.get("predicted_sales", 0.0) for p in predictions)
        avg_pred = total_pred / len(predictions) if predictions else 0.0
        highest_day = max(predictions, key=lambda x: x.get("predicted_sales", 0.0))
        peak_date = highest_day.get("date", "N/A")
        peak_val = highest_day.get("predicted_sales", 0.0)
    else:
        total_pred = float(data.get("total_demand", 0.0))
        avg_pred = float(data.get("avg_daily", total_pred / max(1, horizon)))
        peak_date = "N/A"
        peak_val = avg_pred

    prompt = (
        f"You are a supply chain analyst. Write a concise 2-paragraph business explanation of the sales forecast "
        f"for store '{store_name}'. Use ONLY the verified data provided below. Do not invent any numbers.\n\n"
        f"Forecast Method: {method} (Confidence: {confidence})\n"
        f"Forecast Horizon: {horizon} days\n"
        f"Total Expected Demand: {total_pred:,.1f} units\n"
        f"Average Daily Demand: {avg_pred:,.1f} units\n"
        f"Peak Demand Day: {peak_date} with {peak_val:,.1f} units\n\n"
        f"Focus on operational staffing, replenishment readiness, and peak day management. "
        f"Use professional business language."
    )

    fallback = (
        f"Forecast Analysis for {store_name}: Expected total demand over the next {horizon} days is {total_pred:,.1f} units "
        f"(averaging {avg_pred:,.1f} units/day) with peak day at {peak_val:,.1f} units. "
        f"Model confidence is {confidence} using {method} methodology."
    )

    return await safe_generate_insight(prompt, fallback)


async def explain_inventory_health(store_name_or_id: Any = None, analysis: dict | None = None, **kwargs) -> str:
    """
    Explains inventory signals: stockout risks, excess stock, or healthy coverage.
    """
    store_name = str(store_name_or_id or kwargs.get("store_id") or "Store")
    if not str(store_name).startswith("Store"):
        store_name = f"Store #{store_name}"

    data = analysis or kwargs
    current_qty = float(data.get("current_quantity") or data.get("current_stock") or 0.0)
    days_of_supply = float(data.get("days_of_supply") or 0.0)
    status_label = str(data.get("status") or "unknown")
    demand_rate = float(data.get("average_daily_demand") or data.get("daily_demand_mean") or 0.0)
    lead_time = int(data.get("lead_time_days") or 3)

    prompt = (
        f"You are a procurement advisor. Provide a concise 2-paragraph analysis of inventory conditions for store '{store_name}'. "
        f"Use ONLY the metrics provided below. Do not fabricate figures.\n\n"
        f"Current Inventory: {current_qty:,.1f} units\n"
        f"Average Daily Sales Demand: {demand_rate:,.1f} units/day\n"
        f"Estimated Days of Supply Remaining: {days_of_supply:.1f} days\n"
        f"Supplier Lead Time: {lead_time} days\n"
        f"Inventory Status: {status_label}\n\n"
        f"State whether immediate reorder is required to avoid stockouts or if capital is tied up in excess inventory. "
        f"Provide clear actionable steps for the procurement team."
    )

    fallback = (
        f"Inventory Assessment for {store_name}: Current stock is {current_qty:,.1f} units, representing "
        f"{days_of_supply:.1f} days of supply based on a daily demand of {demand_rate:,.1f} units. "
        f"Current status is evaluated as '{status_label}' with supplier lead time of {lead_time} days."
    )

    return await safe_generate_insight(prompt, fallback)


async def explain_purchase_order(store_name_or_id: Any = None, po_data: dict | None = None, **kwargs) -> str:
    """
    Generates an executive narrative explaining why a specific purchase order quantity
    was recommended.
    """
    store_name = str(store_name_or_id or kwargs.get("store_id") or "Store")
    if not str(store_name).startswith("Store"):
        store_name = f"Store #{store_name}"

    data = po_data or kwargs
    recommended_qty = float(data.get("recommended_qty", 0.0))
    current_inventory = float(data.get("current_inventory", 0.0))
    forecasted_demand = float(data.get("forecasted_demand", 0.0))
    safety_stock = float(data.get("safety_stock", 0.0))
    status_val = str(data.get("status", "pending"))

    if recommended_qty < 0.0 or status_val == "pending_forecast":
        return (
            f"Purchase Order Explanation for {store_name}: Order generation is currently pending "
            f"because reliable forecast data is unavailable. No inventory order should be placed until "
            f"sales demand projections are verified."
        )

    prompt = (
        f"You are a retail procurement strategist. Explain to a store manager why Purchase Order recommendation "
        f"for '{store_name}' was computed. Use ONLY the given figures:\n\n"
        f"Recommended Order Quantity: {recommended_qty:,.1f} units\n"
        f"Current On-Hand Inventory: {current_inventory:,.1f} units\n"
        f"Forecasted Lead-Time Demand: {forecasted_demand:,.1f} units\n"
        f"Safety Stock Buffer: {safety_stock:,.1f} units\n\n"
        f"Explain clearly how (forecasted demand + safety stock - current inventory) yields the recommended quantity. "
        f"Keep explanation under 3 concise paragraphs."
    )

    fallback = (
        f"Purchase Order Recommendation for {store_name}: Recommending {recommended_qty:,.1f} units to replenish "
        f"current stock of {current_inventory:,.1f} units against forecasted demand of {forecasted_demand:,.1f} units "
        f"plus a safety stock buffer of {safety_stock:,.1f} units."
    )

    return await safe_generate_insight(prompt, fallback)
