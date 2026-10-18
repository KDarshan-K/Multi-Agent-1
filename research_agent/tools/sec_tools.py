"""Primary-source SEC EDGAR filing lookup for U.S. public companies with 4h caching."""

from __future__ import annotations

import requests
import config
from tools.cache import cached_call


class SecDataError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not config.SEC_USER_AGENT:
        raise SecDataError("SEC_USER_AGENT is not set in .env (include a contact email).")
    return {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str) -> dict:
    try:
        response = requests.get(url, headers=_headers(), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise SecDataError("Network error calling SEC EDGAR.") from exc
    except ValueError as exc:
        raise SecDataError("SEC EDGAR returned invalid JSON.") from exc


def get_recent_filings(symbol: str, limit: int = 8) -> list[dict]:
    """Return current 10-K, 10-Q, and 8-K links for a U.S.-listed issuer."""
    symbol = symbol.upper().strip()
    cache_key = f"get_recent_filings:{symbol}:{limit}"

    def _fetch():
        ticker_map = _get_json(config.SEC_TICKERS_URL)
        cik = next((int(item["cik_str"]) for item in ticker_map.values()
                    if item.get("ticker", "").upper() == symbol), None)
        if cik is None:
            raise SecDataError(f"SEC EDGAR has no U.S. issuer CIK for {symbol}.")
        recent = _get_json(
            f"{config.SEC_SUBMISSIONS_URL}/CIK{cik:010d}.json"
        ).get("filings", {}).get("recent", {})
        results = []
        for form, accession, date, document in zip(
            recent.get("form", []), recent.get("accessionNumber", []),
            recent.get("filingDate", []), recent.get("primaryDocument", []),
        ):
            if form not in {"10-K", "10-Q", "8-K"} or not document:
                continue
            url = f"{config.SEC_ARCHIVES_URL}/{cik}/{accession.replace('-', '')}/{document}"
            results.append({"symbol": symbol, "form": form, "filing_date": date,
                            "accession_number": accession, "url": url,
                            "source": "SEC EDGAR"})
            if len(results) >= max(1, min(limit, 20)):
                break
        return results

    return cached_call(cache_key, 14400, _fetch)
