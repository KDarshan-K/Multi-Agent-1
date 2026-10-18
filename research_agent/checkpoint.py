"""
Checkpointing (persistence stub).

WHY "STUB": real LangGraph has a proper Checkpointer interface that saves
state after EVERY node execution, letting you resume a run mid-graph
(e.g. after a crash between the fan-out and the analyst). Building that
properly means threading a checkpoint save-call through
graph_engine.py's CompiledGraph.invoke() after every node, plus a
resume-from-node entry point in the graph itself.

This stub implements the simpler, still genuinely useful half: saving the
COMPLETE final state after a run finishes, and reloading it later to
inspect or re-print without rerunning the whole pipeline (see main.py's
`--resume RUN_ID` flag). It does not support resuming mid-graph after a
crash — that's the gap between this and a real checkpointer.

TO GO FURTHER: pass a `checkpoint.save(run_id, state)` call into
graph_engine.CompiledGraph.invoke() right after each node's fn(state)
call, keyed by (run_id, node_name, step_number). Then add a
`resume_from(run_id, node_name)` that reconstructs state and re-enters
build_graph()'s compiled graph at that node instead of the entry point.
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

from state import ResearchState, Evidence

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def _state_to_dict(state: ResearchState) -> dict:
    d = asdict(state)
    # dataclasses.asdict already recurses into the Evidence list correctly
    # since Evidence is itself a dataclass; only `rag_seen_ids` (a set)
    # needs manual conversion since JSON has no set type.
    d["rag_seen_ids"] = sorted(d["rag_seen_ids"])
    return d


def _dict_to_state(d: dict) -> ResearchState:
    evidence = [Evidence(**ev) for ev in d.pop("evidence", [])]
    d["rag_seen_ids"] = set(d.get("rag_seen_ids", []))
    state = ResearchState(query=d.pop("query"))
    for key, value in d.items():
        setattr(state, key, value)
    state.evidence = evidence
    return state


def save(run_id: str, state: ResearchState) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{run_id}.json")
    payload = {
        "run_id": run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": _state_to_dict(state),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load(run_id: str) -> ResearchState:
    path = os.path.join(CHECKPOINT_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found for run_id={run_id!r} at {path}"
        )
    with open(path) as f:
        payload = json.load(f)
    return _dict_to_state(payload["state"])


def list_runs() -> list[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted(
        fname[:-5] for fname in os.listdir(CHECKPOINT_DIR)
        if fname.endswith(".json")
    )
