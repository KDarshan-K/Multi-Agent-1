"""
Search agent node for official LangGraph.
"""

from __future__ import annotations
from state import ResearchState, Evidence
from tools import mcp_client


def search_agent_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    queries = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("web_search:")
    ]

    search_data = []
    evidence = []
    search_queries = []
    data_gaps = []

    for query in queries:
        try:
            results = mcp_client.call_tool("web_search", query=query)
            search_queries.append(query)
            search_data.extend(results)

            for result in results:
                claim = f"{result['title']} — {result['snippet']}"
                evidence.append(Evidence(
                    source="Tavily:web_search",
                    claim=claim,
                    raw_data=result,
                    url=result["url"],
                ))
        except Exception as e:
            data_gaps.append(f"Web search for {query!r} unavailable: {e}")

    return {
        "search_data": search_data,
        "evidence": evidence,
        "search_queries": search_queries,
        "data_gaps": data_gaps,
    }
