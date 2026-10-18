"""
Shared state that flows through every node in the graph.

Compatible with official LangGraph channel reducers (Annotated[list, operator.add]).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import operator
from typing import Annotated, Any, Optional


@dataclass
class Evidence:
    """A single fact pulled from a tool call, tagged with its source."""
    source: str                     # e.g. "MarketTools.get_quote"
    claim: str                      # short human-readable fact
    raw_data: Any = None            # original payload, for debugging/audit
    url: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ResearchState:
    """The state schema passed between nodes in LangGraph."""
    query: str = ""

    # Set by the orchestrator
    plan: list[str] = field(default_factory=list)

    # Accumulated across sub-agents
    market_data: Annotated[list[dict], operator.add] = field(default_factory=list)
    news_data: Annotated[list[dict], operator.add] = field(default_factory=list)
    search_data: Annotated[list[dict], operator.add] = field(default_factory=list)
    rag_context: Annotated[list[dict], operator.add] = field(default_factory=list)

    # Accumulated across every tool call, in source order
    evidence: Annotated[list[Evidence], operator.add] = field(default_factory=list)

    # Set by analyst / critic
    draft_report: str = ""
    critique: Optional[str] = None
    revision_count: int = 0

    # Set by the critic when it wants another round
    next_search_query: Optional[str] = None

    # Set by the critic if MAX_REVISIONS was hit
    unresolved_critique: Optional[str] = None

    # Tracks distinct search queries executed
    search_queries: Annotated[list[str], operator.add] = field(default_factory=list)

    # Dedupe set for RAG retrieval
    rag_seen_ids: set = field(default_factory=set)

    # Set by analyst on fallback synthesis
    synthesis_fallback_reason: Optional[str] = None

    # Tracks live data source failures
    data_gaps: Annotated[list[str], operator.add] = field(default_factory=list)

    # Surfaced disambiguation requests
    unresolved_names: Annotated[list[dict], operator.add] = field(default_factory=list)

    # Final report
    final_report: str = ""

    def add_evidence(self, source: str, claim: str, raw_data: Any = None,
                      url: Optional[str] = None) -> None:
        self.evidence.append(Evidence(source=source, claim=claim,
                                       raw_data=raw_data, url=url))

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchState":
        """Construct a ResearchState object from a dictionary result."""
        evidence_raw = data.get("evidence", [])
        evidence_objects = []
        for ev in evidence_raw:
            if isinstance(ev, Evidence):
                evidence_objects.append(ev)
            elif isinstance(ev, dict):
                evidence_objects.append(Evidence(**ev))

        state = cls(query=data.get("query", ""))
        state.plan = data.get("plan", [])
        state.market_data = data.get("market_data", [])
        state.news_data = data.get("news_data", [])
        state.search_data = data.get("search_data", [])
        state.rag_context = data.get("rag_context", [])
        state.evidence = evidence_objects
        state.draft_report = data.get("draft_report", "")
        state.critique = data.get("critique")
        state.revision_count = data.get("revision_count", 0)
        state.next_search_query = data.get("next_search_query")
        state.unresolved_critique = data.get("unresolved_critique")
        state.search_queries = data.get("search_queries", [])
        state.rag_seen_ids = set(data.get("rag_seen_ids", []))
        state.synthesis_fallback_reason = data.get("synthesis_fallback_reason")
        state.data_gaps = data.get("data_gaps", [])
        state.unresolved_names = data.get("unresolved_names", [])
        state.final_report = data.get("final_report", "")
        return state
