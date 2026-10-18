"""Tavily-backed web search tool with 30m caching."""

from __future__ import annotations

import requests

import config
from tools.cache import cached_call


class WebSearchError(RuntimeError):
    """Raised when Tavily search cannot return trustworthy results."""


def _error_detail(response: requests.Response) -> str:
    """Return a useful API error without ever including an API key."""
    try:
        detail = response.json().get("detail") or response.json().get("error")
    except ValueError:
        detail = response.text
    return str(detail)[:500]


def search(query: str, max_results: int = 3) -> list[dict]:
    """Search the open web with Tavily and return normalized results (cached 30m)."""
    clean_q = query.strip()
    if not clean_q:
        return []
    cache_key = f"web_search:{clean_q.lower()}:{max_results}"

    def _fetch():
        if not config.TAVILY_API_KEY:
            raise WebSearchError(
                "TAVILY_API_KEY is not set. Add it to .env before running a "
                "research report with live web sources."
            )

        payload = {
            "query": clean_q,
            "search_depth": "basic",
            "max_results": max(1, min(max_results, 20)),
            "include_answer": False,
        }
        headers = {
            "Authorization": f"Bearer {config.TAVILY_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                config.TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30
            )
        except requests.RequestException as exc:
            raise WebSearchError(f"Network error calling Tavily: {exc}") from exc

        if response.status_code != 200:
            raise WebSearchError(
                f"Tavily returned {response.status_code}: {_error_detail(response)}"
            )

        results = []
        for item in response.json().get("results", []):
            url = item.get("url")
            title = item.get("title")
            if not url or not title:
                continue
            results.append({
                "title": title,
                "snippet": item.get("content", ""),
                "url": url,
                "source": "Tavily",
            })
        return results

    return cached_call(cache_key, 1800, _fetch)
