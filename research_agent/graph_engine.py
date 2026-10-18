"""
A minimal, stdlib-only stand-in for LangGraph's StateGraph.

WHY THIS EXISTS: this sandbox has no internet access, so `pip install
langgraph` isn't possible here. This module implements just enough of
LangGraph's public API (add_node, add_edge, add_conditional_edges,
set_entry_point, compile().invoke()) to run the real pipeline logic.

TO SWITCH TO REAL LANGGRAPH (recommended once you're off this sandbox):
    pip install langgraph
    - replace `from graph_engine import StateGraph, END`
      with     `from langgraph.graph import StateGraph, END`
    - StateGraph in real LangGraph takes a schema (TypedDict/dataclass) in
      its constructor and nodes return partial state *updates* (dicts) that
      get merged, rather than mutating the state object in place. The node
      functions in this project already return the mutated state object,
      so the only change needed is a thin wrapper if you want to adopt the
      "return a dict of updates" style. Everything else (node names, edges,
      conditional routing) ports as-is.

This engine intentionally supports only what this project needs:
linear edges, fan-out (multiple edges from one node), and one conditional
edge. No parallel execution, no checkpointing (see README for the phase
that adds a checkpointing stub).
"""

from __future__ import annotations
from typing import Callable, Any

END = "__END__"


class StateGraph:
    def __init__(self, state_type: type):
        self.state_type = state_type
        self.nodes: dict[str, Callable[[Any], Any]] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional_edges: dict[str, tuple[Callable, dict[str, str]]] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, fn: Callable[[Any], Any]) -> None:
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges.setdefault(from_node, []).append(to_node)

    def add_conditional_edges(self, from_node: str,
                               router: Callable[[Any], str],
                               mapping: dict[str, str]) -> None:
        self.conditional_edges[from_node] = (router, mapping)

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def compile(self) -> "CompiledGraph":
        if self.entry_point is None:
            raise ValueError("No entry point set. Call set_entry_point() first.")
        return CompiledGraph(self)


class CompiledGraph:
    """Executes the graph breadth-first from the entry point.

    Fan-out (multiple plain edges from one node) is executed sequentially
    in the order added — this stand-in does not thread/async parallelize.
    That's a known simplification vs. real LangGraph; noted in README.
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, state: Any, max_steps: int = 50, verbose: bool = True) -> Any:
        current_nodes = [self.graph.entry_point]
        steps = 0

        while current_nodes:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    f"Graph exceeded max_steps={max_steps}; "
                    "likely an unterminated conditional loop."
                )

            next_nodes: list[str] = []

            for node_name in current_nodes:
                if node_name == END:
                    continue

                fn = self.graph.nodes.get(node_name)
                if fn is None:
                    raise ValueError(f"Unknown node: {node_name}")

                if verbose:
                    print(f"  [graph] running node: {node_name}")
                state = fn(state)

                # Conditional edge takes priority if defined for this node
                if node_name in self.graph.conditional_edges:
                    router, mapping = self.graph.conditional_edges[node_name]
                    branch_key = router(state)
                    target = mapping.get(branch_key)
                    if target is None:
                        raise ValueError(
                            f"Router for '{node_name}' returned unmapped "
                            f"key '{branch_key}'"
                        )
                    if verbose:
                        print(f"  [graph] conditional route -> {branch_key} -> {target}")
                    next_nodes.append(target)
                else:
                    for target in self.graph.edges.get(node_name, []):
                        next_nodes.append(target)

            # de-dupe while preserving order (simple fan-in guard)
            seen = set()
            deduped = []
            for n in next_nodes:
                if n not in seen:
                    seen.add(n)
                    deduped.append(n)
            current_nodes = deduped

        return state
