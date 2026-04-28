from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.config import settings
from src.data_sources import APIError, AlphaVantageClient, NewsAPIClient
from src.model import save_model, train_model


def run_training_pipeline(base_currency: str, quote_currency: str) -> dict[str, Any]:
    if not settings.alpha_vantage_api_key:
        raise APIError("Missing ALPHA_VANTAGE_API_KEY. Add it to your environment before training.")

    fx_client = AlphaVantageClient(api_key=settings.alpha_vantage_api_key)
    prices = fx_client.get_fx_daily(base_currency, quote_currency)
    model, training_result = train_model(prices)

    pair = f"{base_currency}/{quote_currency}"

    # MVP: Skip NewsAPI if key is missing or call fails
    market_context = {"sentiment_score": None, "headline_count": 0, "headlines": []}
    if settings.newsapi_key:
        try:
            news_client = NewsAPIClient(api_key=settings.newsapi_key)
            market_context = news_client.get_market_sentiment(f"{base_currency} {quote_currency} forex")
        except Exception:
            # Log or print error in real app; for MVP just skip
            market_context = {"sentiment_score": None, "headline_count": 0, "headlines": []}

    payload = {
        "pair": pair,
        "training": asdict(training_result),
        "market_context": market_context,
    }
    save_model(model, payload)
    return payload
