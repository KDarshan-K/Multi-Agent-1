"""
Analyst node — the real synthesis step.

Replaces the phase 1-4 template-based report_writer as the DRAFT writer.
Builds a numbered evidence list, asks the LLM to synthesize it into prose
with inline [N] citations tied to that numbering, and stores the result in
state.draft_report. report_writer.py (kept, now a lighter "finalize" step)
appends a references appendix mapping [N] back to sources/URLs.

FALLBACK BEHAVIOR: if no API key is set for the selected provider
(config.LLM_PROVIDER), or the API call fails
for any reason (auth, rate limit, network), this catches it and falls back
to a template-based synthesis so the pipeline never crashes for lack of an
API key. The fallback is clearly labeled in the output so it's never
mistaken for a real LLM synthesis.
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
    """Deterministic, template-based synthesis — no LLM call.

    This is a fallback, not the primary path. It groups evidence by
    source type and writes plain sentences per group, still with [N]
    citation numbers so the finalize step's references appendix behaves
    identically regardless of which path produced the draft.
    """
    lines = [
        "**[FALLBACK MODE — no LLM call made; see notes at the end]**",
        "",
        f"This report addresses: {state.query}",
        "",
    ]
    for i, ev in enumerate(state.evidence, start=1):
        lines.append(f"- {ev.claim} [{i}]")
    return "\n".join(lines)


def analyst_node(state: ResearchState) -> ResearchState:
    if not state.evidence:
        state.draft_report = (
            f"No evidence was gathered for the query: {state.query}"
        )
        return state

    user_prompt = _build_user_prompt(state)

    try:
        synthesis = llm.call_llm(_SYSTEM_PROMPT, user_prompt)
        state.draft_report = synthesis
    except (llm.LLMNotConfiguredError, llm.LLMCallError) as e:
        state.draft_report = _fallback_synthesis(state)
        state.synthesis_fallback_reason = str(e)

    return state
