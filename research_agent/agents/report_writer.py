"""
Report finalizer (formerly the phase 1-4 template report writer).

As of phase 5, agents/analyst.py owns drafting the actual prose (via LLM,
with a template fallback). This node's job shrank to: take that draft,
append a references appendix mapping each [N] citation back to its
source/URL, and add a notes footer. Kept as a separate node (rather than
folding into analyst.py) so citation formatting stays independent of
synthesis — useful once phase 6's critic needs to re-run finalize after a
revision without re-triggering a new LLM call for the same draft.
"""

from __future__ import annotations
from datetime import datetime, timezone
from state import ResearchState
import config


def _references_section(state: ResearchState) -> list[str]:
    lines = ["## References"]
    if not state.evidence:
        lines.append("- None.")
    else:
        for i, ev in enumerate(state.evidence, start=1):
            link = f" — {ev.url}" if ev.url else ""
            lines.append(f"[{i}] {ev.source}{link}")
    lines.append("")
    return lines


def _disambiguation_section(state: ResearchState) -> list[str]:
    """Surfaces companies the orchestrator couldn't confidently resolve
    to a single ticker, so the person can rerun with an explicit symbol
    instead of getting a report that silently guessed (or silently
    skipped) the wrong company."""
    if not state.unresolved_names:
        return []

    lines = ["## ⚠️ Needs disambiguation", ""]
    for entry in state.unresolved_names:
        fragment = entry["query_fragment"]
        candidates = entry["candidates"]
        if not candidates:
            lines.append(
                f"- Couldn't find any ticker matching **{fragment!r}**. "
                f"Try rephrasing or provide the ticker directly."
            )
        else:
            lines.append(
                f"- **{fragment!r}** matched multiple possible companies "
                f"— rerun with the specific ticker you meant:"
            )
            for c in candidates:
                lines.append(
                    f"    - `{c['symbol']}` — {c['name']} "
                    f"({c['region']}, {c['type']})"
                )
    lines.append("")
    return lines


def report_writer_node(state: ResearchState) -> ResearchState:
    lines = [
        f"# Research Report: {state.query}",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]
    lines += _disambiguation_section(state)
    lines += [
        state.draft_report,
        "",
    ]
    lines += _references_section(state)

    notes = [
        "## Notes",
        f"- Went through {state.revision_count} critic-triggered revision"
        f"{'s' if state.revision_count != 1 else ''} before finalizing "
        f"(cap: {config.MAX_REVISIONS}).",
    ]
    if state.synthesis_fallback_reason:
        notes.append(
            f"- ⚠️ Synthesis ran in FALLBACK MODE (no LLM call made): "
            f"{state.synthesis_fallback_reason}"
        )
    if state.unresolved_critique:
        notes.append(
            f"- ⚠️ Revision cap reached with an open critic concern: "
            f"{state.unresolved_critique}"
        )
    if state.data_gaps:
        notes.append(
            f"- ⚠️ {len(state.data_gaps)} live data source call"
            f"{'s' if len(state.data_gaps) != 1 else ''} failed and "
            f"{'were' if len(state.data_gaps) != 1 else 'was'} skipped "
            f"rather than crashing the run:"
        )
        for gap in state.data_gaps:
            notes.append(f"    - {gap}")
    lines += notes

    report = "\n".join(lines)
    state.final_report = report
    return state
