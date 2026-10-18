"""Live Finnhub company news with 30m caching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import requests
import config
from tools.cache import cached_call


class NewsDataError(RuntimeError):
    pass


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    symbol = symbol.upper().strip()
    cache_key = f"get_news:{symbol}:{limit}"

    def _fetch():
        if not config.FINNHUB_API_KEY:
            raise NewsDataError("FINNHUB_API_KEY is not set in .env.")
        end = datetime.now(timezone.utc).date()
        try:
            response = requests.get(
                config.FINNHUB_API_URL,
                params={"symbol": symbol,
                        "from": (end - timedelta(days=365)).isoformat(),
                        "to": end.isoformat(), "token": config.FINNHUB_API_KEY},
                timeout=30,
            )
            response.raise_for_status()
            articles = response.json()
        except requests.RequestException as exc:
            raise NewsDataError("Network error calling Finnhub.") from exc
        except ValueError as exc:
            raise NewsDataError("Finnhub returned invalid JSON.") from exc
        if not isinstance(articles, list):
            raise NewsDataError("Finnhub returned an unexpected news response.")
        results = []
        for item in articles[:max(1, min(limit, 50))]:
            if not item.get("headline") or not item.get("url"):
                continue
            results.append({
                "headline": item["headline"], "summary": item.get("summary", ""),
                "source": item.get("source") or "Finnhub", "url": item["url"],
                "published_at": datetime.fromtimestamp(
                    item.get("datetime", 0), tz=timezone.utc).isoformat(),
                "provider": "Finnhub",
            })
        return results

    return cached_call(cache_key, 1800, _fetch)
