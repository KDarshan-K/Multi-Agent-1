"""
News agent node for official LangGraph.
"""

from __future__ import annotations
from state import ResearchState, Evidence
from tools import mcp_client


def news_agent_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_news:")
    ]

    news_data = []
    evidence = []
    data_gaps = []

    for symbol in symbols:
        try:
            articles = mcp_client.call_tool("get_news", symbol=symbol)
            news_data.extend(articles)

            for article in articles:
                claim = f"{article['headline']} — {article['summary']}"
                evidence.append(Evidence(
                    source=f"MCP:get_news via {article['source']}",
                    claim=claim,
                    raw_data=article,
                    url=article["url"],
                ))
        except Exception as e:
            data_gaps.append(f"News for {symbol} unavailable: {e}")

    return {
        "news_data": news_data,
        "evidence": evidence,
        "data_gaps": data_gaps,
    }
