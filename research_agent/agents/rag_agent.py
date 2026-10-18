"""
RAG retrieval node for official LangGraph.
"""

from __future__ import annotations
from state import ResearchState, Evidence
from tools import mcp_client


def rag_retrieval_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    rag_context = []
    evidence = []
    rag_seen_ids = set(state.rag_seen_ids)

    def _run_query(query_text: str) -> None:
        results = mcp_client.call_tool(
            "knowledge_base_search", query=query_text, top_k=2
        )
        for doc in results:
            if doc["id"] in rag_seen_ids:
                continue
            rag_seen_ids.add(doc["id"])
            rag_context.append(doc)
            evidence.append(Evidence(
                source="MCP:knowledge_base_search",
                claim=f"{doc['title']}: {doc['text']}",
                raw_data=doc,
            ))

    _run_query(state.query)

    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]
    for symbol in symbols:
        _run_query(symbol)

    return {
        "rag_context": rag_context,
        "rag_seen_ids": rag_seen_ids,
        "evidence": evidence,
    }
