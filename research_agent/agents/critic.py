"""
Critic node for official LangGraph.
"""

from __future__ import annotations
import json
import re
from state import ResearchState
import llm
import config


def _heuristic_critique(state: ResearchState) -> str | None:
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
    return [
        step.split(":", 1)[1]
        for step in state.plan
        if step.startswith("fetch_market_data:")
    ]


def _pick_follow_up_doc(state: ResearchState) -> dict | None:
    if not state.rag_context:
        return None

    symbols = set(_current_symbols(state))
    tagged_matches = [
        doc for doc in state.rag_context
        if symbols & set(doc.get("tags", []))
    ]
    candidates = tagged_matches or state.rag_context
    return max(candidates, key=lambda d: d.get("score", 0))


def critic_node(state: ResearchState | dict) -> dict:
    if isinstance(state, dict):
        state = ResearchState.from_dict(state)

    gap: str | None = None
    used_llm = False

    if config.active_api_key():
        try:
            gap = _llm_critique(state)
            used_llm = True
        except (llm.LLMNotConfiguredError, llm.LLMCallError, ValueError,
                 json.JSONDecodeError):
            used_llm = False

    if not used_llm:
        gap = _heuristic_critique(state)

    if gap and state.revision_count < config.MAX_REVISIONS:
        follow_up_doc = _pick_follow_up_doc(state)
        next_search = (
            follow_up_doc["title"] if follow_up_doc else f"{state.query} analysis"
        )
        return {
            "critique": gap,
            "next_search_query": next_search,
            "revision_count": state.revision_count + 1,
            "unresolved_critique": None,
        }
    elif gap:
        return {
            "critique": None,
            "unresolved_critique": gap,
        }
    else:
        return {
            "critique": None,
            "unresolved_critique": None,
        }


def should_revise(state: ResearchState | dict) -> str:
    critique = state.critique if isinstance(state, ResearchState) else state.get("critique")
    return "revise" if critique else "done"
