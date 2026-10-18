"""
Critic node.

This is what turns "critic" from a rubber-stamp formatter into a genuine
feedback loop, per the architecture review: it can send the process BACK
to the orchestrator for another round, not just forward to the final
report.

Two critique paths:
    1. LLM-backed (when an API key is configured for the selected
       provider — see config.LLM_PROVIDER): asks the model
       for a strict JSON verdict on whether the draft adequately covers
       the question given the evidence.
    2. Deterministic heuristic fallback (no key, or the LLM call/parse
       fails): requires at least 2 distinct search queries to have been
       run before accepting the report, on the reasoning that a single
       search pass rarely gives broad enough coverage. This is always
       resolvable within one revision round, so the loop is guaranteed
       to terminate even in fallback mode.

Bounded by config.MAX_REVISIONS regardless of which path fired — if the
cap is hit with a gap still open, the node stops asking for revisions and
instead records the gap in state.unresolved_critique, which report_writer
surfaces honestly in the final Notes section rather than hiding it.
"""

from __future__ import annotations
import json
import re
from state import ResearchState
import llm
import config


def _heuristic_critique(state: ResearchState) -> str | None:
    """Deterministic fallback: require search-depth coverage.

    Always resolvable in exactly one revision round, since the
    orchestrator always adds exactly one new distinct query when acting
    on this critique.
    """
    distinct_queries = set(state.search_queries)
    if len(distinct_queries) < 2:
        example = next(iter(distinct_queries), state.query)
        return (
            f"Only one distinct web search query has been run so far "
            f"('{example}'). Broaden search coverage with an additional, "
            f"more specific query before finalizing."
        )
    return None


def _llm_critique(state: ResearchState) -> str | None:
    """Real LLM call asking for a strict JSON sufficiency verdict.

    Raises on any failure (network, auth, bad JSON) so the caller can
    fall back to the heuristic rather than silently treating a broken
    call as "sufficient".
    """
    evidence_list = "\n".join(
        f"[{i}] {ev.claim}" for i, ev in enumerate(state.evidence, start=1)
    )
    system = (
        "You are a strict editor reviewing a financial research report "
        "draft against its supporting evidence list. Respond with ONLY a "
        "JSON object of the exact shape "
        '{"sufficient": true|false, "gap": "<short description of the '
        'single most important gap, or null if sufficient>"}. '
        "No other text before or after the JSON. Flag missing "
        "perspectives, unaddressed parts of the question, or evidence "
        "that contradicts the draft. Be strict but fair."
    )
    user = (
        f"Question: {state.query}\n\n"
        f"Draft report:\n{state.draft_report}\n\n"
        f"Evidence:\n{evidence_list}"
    )
    raw = llm.call_llm(system, user, max_tokens=300)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Critic LLM response wasn't valid JSON: {raw[:200]!r}")
    verdict = json.loads(match.group(0))

    if verdict.get("sufficient", True):
        return None
    return verdict.get("gap") or "Critic flagged an unspecified gap."


def _current_symbols(state: ResearchState) -> list[str]:
    """Re-derive which tickers this round's plan was actually about, so
    the follow-up query can be steered toward company-specific context
    instead of whatever happened to be inserted into rag_context first."""
    return [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]


def _pick_follow_up_doc(state: ResearchState) -> dict | None:
    """Pick the RAG doc to base the follow-up search query on.

    Bug this fixes: naively using state.rag_context[0] picks whichever
    doc happened to be inserted first, which is an artifact of which
    tool call ran first — NOT necessarily the most relevant doc. Caught
    via a real run: a query about Microsoft's profit had a genuine
    TF-IDF score TIE (0.1195) between a generic "analyst price target"
    doc and the actual MSFT-specific doc; list order (not relevance)
    decided the tie, and the generic doc won, producing a low-value
    follow-up search with no company connection at all.

    Fix: prefer a doc tagged with one of the tickers actually being
    researched this round (using the real per-symbol score, which was
    unambiguous — 0.1603 for the correct doc vs. the tied 0.1195 — even
    though the combined/base-query score alone couldn't distinguish
    them). Only fall back to the highest-scoring doc overall, then to
    plain list order, if no tagged match exists.
    """
    if not state.rag_context:
        return None

    symbols = set(_current_symbols(state))
    tagged_matches = [
        doc for doc in state.rag_context
        if symbols & set(doc.get("tags", []))
    ]
    candidates = tagged_matches or state.rag_context
    return max(candidates, key=lambda d: d.get("score", 0))


def critic_node(state: ResearchState) -> ResearchState:
    gap: str | None = None
    used_llm = False

    if config.active_api_key():
        try:
            gap = _llm_critique(state)
            used_llm = True
        except (llm.LLMNotConfiguredError, llm.LLMCallError, ValueError,
                 json.JSONDecodeError):
            used_llm = False  # fall through to heuristic below

    if not used_llm:
        gap = _heuristic_critique(state)

    if gap and state.revision_count < config.MAX_REVISIONS:
        state.critique = gap
        # Use the chosen doc's title as its own standalone follow-up
        # query rather than concatenating it onto the full original
        # query — concatenating produced garbled, overlong search
        # queries in testing (e.g. "give the earnings of amazon in 2025
        # AMZN — AWS margin trends and retail segment context"). A
        # short, independent, company-relevant query works better with
        # the mock search tool's template matching.
        follow_up_doc = _pick_follow_up_doc(state)
        state.next_search_query = (
            follow_up_doc["title"] if follow_up_doc else f"{state.query} analysis"
        )
        state.revision_count += 1
    elif gap:
        # Cap reached but a gap remains — say so, don't hide it.
        state.unresolved_critique = gap
        state.critique = None
    else:
        state.critique = None

    return state


def should_revise(state: ResearchState) -> str:
    """Router for graph.py's conditional edge."""
    return "revise" if state.critique else "done"
