"""
Graph definition using official LangGraph.

Wiring topology:

              ┌─ market_agent ─┐
    orchestrator ─ news_agent  ─┼─→ rag_retrieval → analyst → critic ─┬─ revise → orchestrator (loop)
              └─ search_agent ─┘                                     └─ done   → report_writer → END
"""

from langgraph.graph import StateGraph, END
from state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.market_agent import market_agent_node
from agents.news_agent import news_agent_node
from agents.search_agent import search_agent_node
from agents.rag_agent import rag_retrieval_node
from agents.analyst import analyst_node
from agents.critic import critic_node, should_revise
from agents.report_writer import report_writer_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("market_agent", market_agent_node)
    graph.add_node("news_agent", news_agent_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report_writer", report_writer_node)

    graph.set_entry_point("orchestrator")

    # fan-out
    graph.add_edge("orchestrator", "market_agent")
    graph.add_edge("orchestrator", "news_agent")
    graph.add_edge("orchestrator", "search_agent")

    # converge into rag_retrieval (shared context layer), then synthesize
    graph.add_edge("market_agent", "rag_retrieval")
    graph.add_edge("news_agent", "rag_retrieval")
    graph.add_edge("search_agent", "rag_retrieval")
    graph.add_edge("rag_retrieval", "analyst")
    graph.add_edge("analyst", "critic")

    # THE FEEDBACK LOOP
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {"revise": "orchestrator", "done": "report_writer"},
    )

    graph.add_edge("report_writer", END)

    return graph.compile()
