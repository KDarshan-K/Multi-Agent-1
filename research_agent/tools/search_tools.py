"""Tavily-backed web search tool.

Search is intentionally general-purpose: it accepts free-text queries and
does not contain any company- or ticker-specific logic.  It returns only
the fields the rest of the research pipeline needs: title, snippet, and the
original result URL.

Set ``TAVILY_API_KEY`` in ``.env`` before running the pipeline.  Unlike the
old mock implementation, this tool never fabricates search results when a
key is missing or a request fails.
"""

from __future__ import annotations

import requests

import config


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
    """Search the open web with Tavily and return normalized results."""
    if not config.TAVILY_API_KEY:
        raise WebSearchError(
            "TAVILY_API_KEY is not set. Add it to .env before running a "
            "research report with live web sources."
        )
    if not query.strip():
        return []

    payload = {
        "query": query,
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
