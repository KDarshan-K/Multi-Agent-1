"""Live Alpha Vantage company-profile tool."""

from __future__ import annotations

from tools.market_tools import MarketDataError, _request


def get_company_profile(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    data = _request("OVERVIEW", symbol=symbol)
    if not data.get("Symbol"):
        raise MarketDataError(f"Alpha Vantage returned no company profile for {symbol}.")
    return {
        "symbol": data.get("Symbol", symbol),
        "name": data.get("Name", symbol),
        "sector": data.get("Sector") or "Unclassified",
        "market_cap": int(float(data.get("MarketCapitalization") or 0)),
        "description": data.get("Description") or "No description returned.",
        "source": "Alpha Vantage OVERVIEW",
        "url": "https://www.alphavantage.co/documentation/#company-overview",
    }
