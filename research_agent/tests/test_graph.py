from state import ResearchState
from graph import build_graph


def test_graph_compilation_and_execution():
    app = build_graph()
    assert app is not None

    init_state = ResearchState(query="How is Apple doing today?")
    result = app.invoke(init_state)

    if isinstance(result, dict):
        state = ResearchState.from_dict(result)
    else:
        state = result

    assert state.query == "How is Apple doing today?"
    assert len(state.plan) > 0
    assert state.final_report != ""
    assert "Research Report" in state.final_report
