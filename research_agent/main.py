"""
Entry point.

Usage:
    python main.py "How is Apple and Tesla doing today?"
    python main.py --list-runs
    python main.py --resume <run_id>
"""

from __future__ import annotations
import sys
import uuid
from state import ResearchState
from graph import build_graph
import checkpoint


def run(query: str, verbose: bool = True) -> tuple[str, ResearchState]:
    run_id = uuid.uuid4().hex[:8]
    app = build_graph()
    initial_state = ResearchState(query=query)
    final_state = app.invoke(initial_state, verbose=verbose)
    checkpoint.save(run_id, final_state)
    return run_id, final_state


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--list-runs":
        runs = checkpoint.list_runs()
        if not runs:
            print("No saved runs yet.")
        else:
            print("Saved runs:")
            for run_id in runs:
                print(f"  {run_id}")
        return

    if args and args[0] == "--resume":
        if len(args) < 2:
            print("Usage: python main.py --resume <run_id>")
            sys.exit(1)
        run_id = args[1]
        try:
            state = checkpoint.load(run_id)
        except FileNotFoundError as e:
            print(str(e))
            print("Run `python main.py --list-runs` to see available runs.")
            sys.exit(1)
        print(f"\n[resumed run {run_id} — showing saved final report, "
              f"not re-running the pipeline]\n{'='*60}\n")
        print(state.final_report)
        return

    query = " ".join(args) or "How is Apple and Tesla doing today?"
    print(f"\nQuery: {query}\n{'='*60}")
    run_id, final_state = run(query)
    print(f"{'='*60}\n")
    print(final_state.final_report)
    print(f"\n[checkpoint saved — run id: {run_id}. "
          f"Re-view anytime with: python main.py --resume {run_id}]")


if __name__ == "__main__":
    main()
