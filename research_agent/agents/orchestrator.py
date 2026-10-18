"""
Orchestrator node.

Responsibility: parse the user query for stock ticker symbols and build a
plan that fans out to Market, News, and Search agents on the first pass.

On a revision pass (revision_count > 0, meaning the Critic sent us back),
builds a MINIMAL plan containing only the new work the critic asked for
(state.next_search_query) — not a full re-fetch of everything. This
matters mechanically: market_agent/news_agent filter state.plan for their
own prefix, so if this revision's plan only contains a `web_search:`
entry, those agents just see an empty list and become no-ops for this
round, instead of re-fetching (and duplicating evidence for) data we
already have.

AUTOMATIC TICKER RESOLUTION (new): if a query names a company that isn't
an explicit ticker and isn't in the hardcoded _COMPANY_NAME_TO_TICKER map
(e.g. "Give me a report on Costco"), the orchestrator now calls Alpha
Vantage's SYMBOL_SEARCH (via the same MCP tool layer every other agent
uses) to resolve it automatically, rather than silently skipping
market/news/SEC data for that company. This is a deliberate layering
change: the orchestrator now makes a live tool call, where before it was
pure offline text parsing.

If the resolution is confident (one dominant match), it's used
transparently. If it's genuinely ambiguous (multiple candidates with
comparable relevance — e.g. same name across exchanges, or several
similarly-named companies) or there's no match at all, the orchestrator
does NOT guess: it records the candidates in state.unresolved_names for
the report to surface as a disambiguation request, and skips
market/news/SEC fetching for that name rather than risk reporting on the
wrong company.

COST NOTE: this costs one additional Alpha Vantage call (same 25/day,
5/minute quota as get_quote/get_company_profile) per unresolved name — so
it's only attempted when a name genuinely can't be resolved offline
first, to avoid burning quota unnecessarily.
"""

from __future__ import annotations
import re
from state import ResearchState
from tools import mcp_client

# Company-name fallback: catches the very common case of a query naming a
# company in plain English ("give me Amazon's earnings") rather than its
# ticker symbol. Checked as a substring match against the lowercased
# query. Still not real NER — a small hardcoded lookup — but it resolves
# the most common cases (and the most-searched companies) with zero
# network cost, before falling back to a live SYMBOL_SEARCH call for
# anything not in this list.
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
}

# Words stripped out when deriving a company-name candidate from the raw
# query text, before sending what's left to SYMBOL_SEARCH. Deliberately
# generous (covers common query scaffolding and generic finance terms) to
# cut down on wasted Alpha Vantage calls for queries that don't actually
# name a specific company — but this is a heuristic, not real NLP, and
# will occasionally over- or under-strip on an unusual phrasing.
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
    "has", "have", "had", "with", "from", "by", "as", "it",
}

# Confidence thresholds for auto-resolving a SYMBOL_SEARCH result rather
# than treating it as ambiguous. Alpha Vantage's matchScore is 0-1; a
# well-known company's correctly-spelled name typically scores close to
# 1.0 with a clear gap to the next-best match. These were chosen to be
# conservative — prefer surfacing an ambiguous case over guessing wrong.
_CONFIDENT_SCORE = 0.8
_MIN_SCORE_GAP = 0.15
_MAX_CANDIDATES_SHOWN = 5


# Common all-caps abbreviations that are NOT tickers, but match the same
# 1-5 letter uppercase pattern real tickers do. Without this, something
# like "how did nvidia do in FY 2026" would treat "FY" as a candidate
# ticker, wasting an Alpha Vantage call (and burning quota) on a fiscal-
# year abbreviation rather than a real symbol. Not exhaustive — just the
# ones likely to show up in a finance-flavored query.
_NON_TICKER_ABBREVIATIONS = {
    "FY", "Q1", "Q2", "Q3", "Q4", "CEO", "CFO", "COO", "CTO", "IPO",
    "ETF", "GDP", "USD", "EPS", "YOY", "QOQ", "SEC", "US", "USA", "UK",
    "EU", "AI", "IT",
}


def _extract_symbols(query: str) -> list[str]:
    # Keep only all-uppercase tokens from the original query. This permits
    # any explicit ticker (e.g. IBM, PLTR, BRK) without turning ordinary
    # query words such as "give" into fake symbols.
    found = set(re.findall(r"\b[A-Z]{1,5}\b", query)) - _NON_TICKER_ABBREVIATIONS

    query_lower = query.lower()
    for name, ticker in _COMPANY_NAME_TO_TICKER.items():
        if name in query_lower:
            found.add(ticker)

    return sorted(found)


def _candidate_company_name(query: str) -> str:
    """Best-effort extraction of a company-name candidate from free text,
    for when no ticker or known company name was already found. Not real
    NLP — just strips common query scaffolding and generic finance words,
    leaving whatever's left (ideally just the company name)."""
    tokens = re.findall(r"[A-Za-z]+", query)
    kept = [t for t in tokens if t.lower() not in _FILLER_TOKENS]
    return " ".join(kept).strip()


def _resolve_via_symbol_search(query: str, state: ResearchState) -> list[str]:
    """Try to resolve a free-text company name to a ticker via Alpha
    Vantage SYMBOL_SEARCH. Returns a list containing the resolved ticker
    if confident, or an empty list if unresolved/ambiguous/failed — in
    which case state.unresolved_names or state.data_gaps is populated so
    the report can explain what happened rather than silently doing
    nothing.
    """
    candidate = _candidate_company_name(query)
    if len(candidate) < 2:
        return []  # nothing meaningful to search; don't waste a call

    try:
        matches = mcp_client.call_tool("search_symbol", keywords=candidate)
    except Exception as e:
        state.data_gaps.append(
            f"Ticker resolution for {candidate!r} unavailable: {e}"
        )
        return []

    if not matches:
        state.unresolved_names.append({
            "query_fragment": candidate,
            "candidates": [],
        })
        return []

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    top = matches[0]
    second_score = matches[1]["match_score"] if len(matches) > 1 else 0.0

    confident = (
        top["match_score"] >= _CONFIDENT_SCORE
        and (top["match_score"] - second_score) >= _MIN_SCORE_GAP
    )
    if confident:
        return [top["symbol"]]

    # Ambiguous — surface candidates rather than guess.
    state.unresolved_names.append({
        "query_fragment": candidate,
        "candidates": matches[:_MAX_CANDIDATES_SHOWN],
    })
    return []


def orchestrator_node(state: ResearchState) -> ResearchState:
    if state.revision_count == 0:
        # first pass: build the initial plan, fanning out to all 3 agents
        symbols = _extract_symbols(state.query)

        if not symbols:
            # No ticker, no known company name — try live resolution
            # before giving up on structured data for this query.
            symbols = _resolve_via_symbol_search(state.query, state)

        plan = []
        for s in symbols:
            plan.append(f"fetch_market_data:{s}")
            plan.append(f"fetch_news:{s}")
        plan.append(f"web_search:{state.query}")
        state.plan = plan
    else:
        # Revision pass: minimal plan, just the critic's follow-up query.
        plan = []
        if state.next_search_query:
            plan.append(f"web_search:{state.next_search_query}")
            state.next_search_query = None
        state.plan = plan

    return state
