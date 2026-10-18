"""
RAG retrieval node.

Deliberately its own node, sitting AFTER the market/news/search fan-out
converges and BEFORE the analyst — not nested under any single
sub-agent. This directly addresses a critique of the original
architecture diagram, where the RAG/Knowledge box was drawn as a child of
only the "Company Data" branch, making it invisible to the News/Search
agents' findings.

Queries the knowledge base once with the overall user query, and once
more per extracted ticker symbol (pulled back out of the plan) so it picks
up symbol-specific context even if the user's original phrasing didn't
mention the ticker explicitly.

Dedupe uses state.rag_seen_ids (persistent across the whole run) rather
than a set local to one node call — this node re-runs on every critic
revision loop, and without state-level dedup it would re-add the same
knowledge-base doc as duplicate evidence on every round.
"""

from __future__ import annotations
from state import ResearchState
from tools import mcp_client


def rag_retrieval_node(state: ResearchState) -> ResearchState:
    def _run_query(query_text: str) -> None:
        results = mcp_client.call_tool(
            "knowledge_base_search", query=query_text, top_k=2
        )
        for doc in results:
            if doc["id"] in state.rag_seen_ids:
                continue  # already added in this run (or an earlier revision round)
            state.rag_seen_ids.add(doc["id"])
            state.rag_context.append(doc)
            state.add_evidence(
                source="MCP:knowledge_base_search",
                claim=f"{doc['title']}: {doc['text']}",
                raw_data=doc,
            )

    _run_query(state.query)

    symbols = [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]
    for symbol in symbols:
        _run_query(symbol)

    return state
