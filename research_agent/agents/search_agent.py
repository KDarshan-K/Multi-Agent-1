"""
Search agent node.

Reads `web_search:<QUERY>` entries from the plan, calls the search tool via
the MCP client for each, stores raw results in state.search_data, and logs
each result as Evidence.

Also records every distinct query executed into state.search_queries —
this is what the Critic node (phase 6) checks to decide whether search
coverage is broad enough, or whether to send the orchestrator back for
another round with a different query.

Calls tools through tools/mcp_client.py (the unified MCP-style layer) —
never imports search_tools directly.

Wrapped in try/except: this now hits real Tavily API calls. A query is
only appended to state.search_queries on SUCCESS — not on every attempt —
so the Critic's coverage check (which counts distinct successful
searches) doesn't get fooled into declaring "sufficient" based on queries
that actually returned nothing. A failure is recorded in state.data_gaps
instead, and the pipeline continues rather than crashing.
"""

from __future__ import annotations
from state import ResearchState
from tools import mcp_client


def search_agent_node(state: ResearchState) -> ResearchState:
    queries = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("web_search:")
    ]

    for query in queries:
        try:
            results = mcp_client.call_tool("web_search", query=query)
            state.search_queries.append(query)
            state.search_data.extend(results)

            for result in results:
                claim = f"{result['title']} — {result['snippet']}"
                state.add_evidence(
                    source="Tavily:web_search",
                    claim=claim,
                    raw_data=result,
                    url=result["url"],
                )
        except Exception as e:
            state.data_gaps.append(f"Web search for {query!r} unavailable: {e}")

    return state
