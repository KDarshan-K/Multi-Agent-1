"""
Orchestrator node for official LangGraph.
"""

from __future__ import annotations
import re
from state import ResearchState
from tools import mcp_client

_COMPANY_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "costco": "COST",
    "intel": "INTC",
    "walmart": "WMT",
    "disney": "DIS",
    "berkshire": "BRK.B",
    "jpmorgan": "JPM",
    "broadcom": "AVGO",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
    "palantir": "PLTR",
    "boeing": "BA",
    "uber": "UBER",
    "airbnb": "ABNB",
    "qualcomm": "QCOM",
    "coinbase": "COIN",
    "spotify": "SPOT",
}

_FILLER_TOKENS = {
    "give", "me", "a", "an", "the", "report", "on", "how", "did", "do",
    "does", "is", "are", "in", "for", "what", "tell", "about", "earnings",
    "profit", "doing", "performance", "of", "fy", "q1", "q2", "q3", "q4",
    "and", "vs", "versus", "compare", "comparing", "stock", "stocks",
    "market", "today", "year", "years", "company", "companies", "best",
    "top", "current", "currently", "now", "recent", "recently", "news",
    "data", "information", "summary", "overview", "analysis", "please",
    "can", "you", "i", "want", "need", "to", "get", "show", "see",
    "this", "that", "these", "those", "its", "their", "was", "were",
    "has", "have", "had", "with", "from", "by", "as", "it", "margins",
}

_CONFIDENT_SCORE = 0.8
_MIN_SCORE_GAP = 0.15
_MAX_CANDIDATES_SHOWN = 5

_NON_TICKER_ABBREVIATIONS = {
    "FY", "Q1", "Q2", "Q3", "Q4", "CEO", "CFO", "COO", "CTO", "IPO",
    "ETF", "GDP", "USD", "EPS", "YOY", "QOQ", "SEC", "US", "USA", "UK",
    "EU", "AI", "IT",
}


def _extract_symbols(query: str) -> list[str]:
    found = set(re.findall(r"\b[A-Z]{1,5}\b", query)) - _NON_TICKER_ABBREVIATIONS
    query_clean = re.sub(r"['’]s\b", "", query.lower())
    for name, ticker in _COMPANY_NAME_TO_TICKER.items():
        if re.search(r"\b" + re.escape(name) + r"\b", query_clean):
            found.add(ticker)
    return sorted(found)


def _candidate_company_name(query: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", query)
    kept = [t for t in tokens if t.lower() not in _FILLER_TOKENS]
    return " ".join(kept).strip()


def _resolve_via_symbol_search(query: str, state: ResearchState) -> tuple[list[str], list[str], list[dict]]:
    candidate = _candidate_company_name(query)
    if len(candidate) < 2:
        return [], [], []

    try:
        matches = mcp_client.call_tool("search_symbol", keywords=candidate)
    except Exception as e:
        return [], [f"Ticker resolution for {candidate!r} unavailable: {e}"], []

    if not matches:
        return [], [], [{"query_fragment": candidate, "candidates": []}]

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    top = matches[0]
    second_score = matches[1]["match_score"] if len(matches) > 1 else 0.0

    confident = (
        top["match_score"] >= _CONFIDENT_SCORE
        and (top["match_score"] - second_score) >= _MIN_SCORE_GAP
    )
    if confident:
        return [top["symbol"]], [], []

    return [], [], [{"query_fragment": candidate, "candidates": matches[:_MAX_CANDIDATES_SHOWN]}]


def orchestrator_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    data_gaps = []
    unresolved_names = []

    if state.revision_count == 0:
        symbols = _extract_symbols(state.query)
        if not symbols:
            symbols, gaps, unres = _resolve_via_symbol_search(state.query, state)
            data_gaps.extend(gaps)
            unresolved_names.extend(unres)

        plan = []
        for s in symbols:
            plan.append(f"fetch_market_data:{s}")
            plan.append(f"fetch_news:{s}")
        plan.append(f"web_search:{state.query}")
    else:
        plan = []
        if state.next_search_query:
            plan.append(f"web_search:{state.next_search_query}")

    return {
        "plan": plan,
        "next_search_query": None,
        "data_gaps": data_gaps,
        "unresolved_names": unresolved_names,
    }
