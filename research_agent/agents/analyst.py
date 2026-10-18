"""
Analyst node for official LangGraph.
"""

from __future__ import annotations
from state import ResearchState
import llm

_SYSTEM_PROMPT = (
    "You are a financial research analyst. You will be given a user's "
    "research question and a numbered list of evidence items gathered by "
    "other agents (market data, news, web search, and internal knowledge "
    "base context). Write a clear, well-organized report that answers the "
    "question using ONLY the evidence provided — do not invent facts not "
    "present in the evidence list. Cite every factual claim with the "
    "matching evidence number in square brackets, e.g. [3]. If evidence "
    "items conflict, note the discrepancy rather than picking one "
    "silently. Structure the report with short section headers. Do not "
    "list the evidence itself verbatim — synthesize it into analysis."
)


def _build_user_prompt(state: ResearchState) -> str:
    lines = [f"Research question: {state.query}", "", "Evidence:"]
    for i, ev in enumerate(state.evidence, start=1):
        lines.append(f"[{i}] {ev.claim}")
    return "\n".join(lines)


def _fallback_synthesis(state: ResearchState) -> str:
    lines = [
        "**[FALLBACK MODE — no LLM call made; see notes at the end]**",
        "",
        f"This report addresses: {state.query}",
        "",
    ]
    for i, ev in enumerate(state.evidence, start=1):
        lines.append(f"- {ev.claim} [{i}]")
    return "\n".join(lines)


def analyst_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    if not state.evidence:
        return {"draft_report": f"No evidence was gathered for the query: {state.query}"}

    user_prompt = _build_user_prompt(state)

    try:
        synthesis = llm.call_llm(_SYSTEM_PROMPT, user_prompt)
        return {"draft_report": synthesis}
    except (llm.LLMNotConfiguredError, llm.LLMCallError) as e:
        return {
            "draft_report": _fallback_synthesis(state),
            "synthesis_fallback_reason": str(e),
        }
