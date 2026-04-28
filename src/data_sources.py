from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


class APIError(RuntimeError):
    """Raised when an external API request fails."""


@dataclass
class AlphaVantageClient:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"

    def get_fx_daily(self, from_symbol: str, to_symbol: str, outputsize: str = "full") -> pd.DataFrame:
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if "Error Message" in payload:
            raise APIError(payload["Error Message"])
        if "Note" in payload:
            raise APIError(payload["Note"])

        series = payload.get("Time Series FX (Daily)")
        if not series:
            raise APIError("Alpha Vantage did not return FX daily data.")

        df = pd.DataFrame.from_dict(series, orient="index").rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
            }
        )
        df.index = pd.to_datetime(df.index)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.sort_index()


@dataclass
class NewsAPIClient:
    api_key: str
    base_url: str = "https://newsapi.org/v2/everything"

    def get_market_sentiment(self, query: str, page_size: int = 20) -> dict[str, Any]:
        if not self.api_key:
            return {
                "sentiment_score": 0.0,
                "headline_count": 0,
                "headlines": [],
            }

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            raise APIError(payload.get("message", "NewsAPI request failed."))

        articles = payload.get("articles", [])
        positive_terms = {"gain", "bullish", "surge", "beat", "growth", "strong", "optimism"}
        negative_terms = {"drop", "bearish", "slump", "miss", "weak", "risk", "fear"}

        score = 0
        headlines: list[str] = []
        for article in articles:
            title = (article.get("title") or "").lower()
            headlines.append(article.get("title") or "")
            score += sum(term in title for term in positive_terms)
            score -= sum(term in title for term in negative_terms)

        normalized = score / max(len(articles), 1)
        return {
            "sentiment_score": round(normalized, 4),
            "headline_count": len(articles),
            "headlines": headlines[:5],
        }
