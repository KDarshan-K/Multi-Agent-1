"""Live Alpha Vantage market-data tools; no mock prices are produced."""

from __future__ import annotations

import requests
import config


class MarketDataError(RuntimeError):
    pass


def _request(function: str, **params: str) -> dict:
    if not config.ALPHA_VANTAGE_API_KEY:
        raise MarketDataError("ALPHA_VANTAGE_API_KEY is not set in .env.")
    try:
        response = requests.get(
            config.ALPHA_VANTAGE_API_URL,
            params={"function": function, "apikey": config.ALPHA_VANTAGE_API_KEY,
                    **params},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise MarketDataError("Network error calling Alpha Vantage.") from exc
    except ValueError as exc:
        raise MarketDataError("Alpha Vantage returned invalid JSON.") from exc
    message = data.get("Error Message") or data.get("Information") or data.get("Note")
    if message:
        raise MarketDataError(f"Alpha Vantage: {message}")
    return data


def get_quote(symbol: str) -> dict:
    """Return the latest provider quote for a ticker symbol."""
    symbol = symbol.upper().strip()
    quote = _request("GLOBAL_QUOTE", symbol=symbol).get("Global Quote", {})
    if not quote.get("05. price"):
        raise MarketDataError(f"Alpha Vantage returned no quote for {symbol}.")
    return {
        "symbol": quote.get("01. symbol", symbol),
        "price": float(quote["05. price"]),
        "change_pct": float(quote.get("10. change percent", "0").rstrip("%")),
        "volume": int(float(quote.get("06. volume", 0))),
        "as_of": quote.get("07. latest trading day"),
        "source": "Alpha Vantage GLOBAL_QUOTE",
        "url": "https://www.alphavantage.co/documentation/#latestprice",
    }


def get_quotes(symbols: list[str]) -> list[dict]:
    return [get_quote(symbol) for symbol in symbols]


def search_symbol(keywords: str) -> list[dict]:
    """Resolve a free-text company name to candidate ticker symbols.

    Uses Alpha Vantage's SYMBOL_SEARCH endpoint. Returns Alpha Vantage's
    own relevance ranking (matchScore, 0-1) so callers (orchestrator.py)
    can decide whether the top result is confident enough to auto-select,
    or whether multiple candidates are close enough that the choice
    should be surfaced to the person instead of guessed.

    Costs one Alpha Vantage call against the same 25/day, 5/minute quota
    as get_quote/get_company_profile — orchestrator.py only calls this
    when a name can't already be resolved via ticker regex or the
    hardcoded company map, to avoid spending quota when it isn't needed.
    """
    data = _request("SYMBOL_SEARCH", keywords=keywords)
    matches = data.get("bestMatches", [])
    return [
        {
            "symbol": m.get("1. symbol", ""),
            "name": m.get("2. name", ""),
            "type": m.get("3. type", ""),
            "region": m.get("4. region", ""),
            "currency": m.get("8. currency", ""),
            "match_score": float(m.get("9. matchScore", 0) or 0),
        }
        for m in matches
        if m.get("1. symbol")
    ]
