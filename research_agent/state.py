"""
Shared state that flows through every node in the graph.

Uses plain dataclasses instead of pydantic (this sandbox has no network
access to pip-install pydantic). Field names/shape are kept pydantic-friendly
on purpose — swapping to a BaseModel later is a direct 1:1 port.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Evidence:
    """A single fact pulled from a tool call, tagged with its source.

    This is what lets the final report cite where every claim came from,
    and lets the Critic (added in a later phase) check for unsupported
    claims.
    """
    source: str                     # e.g. "MarketTools.get_quote"
    claim: str                      # short human-readable fact
    raw_data: Any = None            # original payload, for debugging/audit
    url: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ResearchState:
    """The single object passed between every node in the graph.

    Phase 1 only populates: query, plan, market_data, evidence, draft_report,
    final_report. Later phases add news_data, search_data, rag_context,
    critique, revision_count without breaking this shape.
    """
    query: str

    # Set by the orchestrator
    plan: list[str] = field(default_factory=list)

    # Set by sub-agents
    market_data: list[dict] = field(default_factory=list)
    news_data: list[dict] = field(default_factory=list)
    search_data: list[dict] = field(default_factory=list)
    rag_context: list[dict] = field(default_factory=list)

    # Accumulated across every tool call, in source order
    evidence: list[Evidence] = field(default_factory=list)

    # Set by analyst / critic
    draft_report: str = ""
    critique: Optional[str] = None
    revision_count: int = 0

    # Set by the critic when it wants another round: a concrete follow-up
    # search query for the orchestrator to act on. Cleared once consumed.
    next_search_query: Optional[str] = None

    # Set by the critic if MAX_REVISIONS was hit while a gap still
    # remained — surfaced honestly in the final report rather than hidden.
    unresolved_critique: Optional[str] = None

    # Tracks every distinct search query executed so far (across possibly
    # multiple revision rounds) — used by the critic's coverage check.
    search_queries: list[str] = field(default_factory=list)

    # Dedupe set for RAG retrieval, kept at state level (not local to one
    # node call) so it persists correctly across revision-loop re-runs of
    # rag_retrieval_node.
    rag_seen_ids: set = field(default_factory=set)

    # Set by the analyst if it had to fall back to template synthesis
    # (e.g. no API key). Kept separate from `critique`, which is reserved
    # for the Critic node's actual quality assessment.
    synthesis_fallback_reason: Optional[str] = None

    # Tracks any live data source that failed this run (rate limit,
    # missing key, network error, bad ticker, etc.) — so a single failed
    # API call degrades gracefully instead of crashing the whole pipeline.
    # Surfaced honestly in the final report's Notes rather than hidden.
    data_gaps: list = field(default_factory=list)

    # Set by the orchestrator when a company name couldn't be confidently
    # resolved to one ticker via search_symbol — either zero matches, or
    # several candidates close enough in relevance that guessing would be
    # irresponsible. Each entry: {"query_fragment": str, "candidates":
    # list[dict]}. Surfaced in the report as a disambiguation request
    # instead of silently picking one and possibly reporting on the
    # wrong company.
    unresolved_names: list = field(default_factory=list)

    # Final output
    final_report: str = ""

    def add_evidence(self, source: str, claim: str, raw_data: Any = None,
                      url: Optional[str] = None) -> None:
        self.evidence.append(Evidence(source=source, claim=claim,
                                       raw_data=raw_data, url=url))
