"""
News agent node.

Reads `fetch_news:<SYMBOL>` entries from the plan, calls the news tool via
the MCP client for each, stores raw articles in state.news_data, and logs
each headline as Evidence.

Calls tools through tools/mcp_client.py (the unified MCP-style layer) —
never imports news_tools directly.

Wrapped in try/except: this now hits real Finnhub API calls, and a
missing key / rate limit / network error shouldn't crash the whole
pipeline — it's recorded in state.data_gaps and the run continues with
whatever other data was gathered successfully.
"""

from __future__ import annotations
from state import ResearchState
from tools import mcp_client


def news_agent_node(state: ResearchState) -> ResearchState:
    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_news:")
    ]

    for symbol in symbols:
        try:
            articles = mcp_client.call_tool("get_news", symbol=symbol)
            state.news_data.extend(articles)

            for article in articles:
                claim = f"{article['headline']} — {article['summary']}"
                state.add_evidence(
                    source=f"MCP:get_news via {article['source']}",
                    claim=claim,
                    raw_data=article,
                    url=article["url"],
                )
        except Exception as e:
            state.data_gaps.append(f"News for {symbol} unavailable: {e}")

    return state
