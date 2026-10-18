from state import ResearchState
from agents.critic import critic_node, should_revise


def test_heuristic_critique_triggers_revision():
    state = ResearchState(query="Tell me about Apple")
    state.search_queries = ["Tell me about Apple"]
    state.revision_count = 0

    state_dict = critic_node(state)
    assert state_dict["critique"] is not None
    assert state_dict["revision_count"] == 1

    route = should_revise(state_dict)
    assert route == "revise"


def test_heuristic_critique_completes_after_two_queries():
    state = ResearchState(query="Tell me about Apple")
    state.search_queries = ["Tell me about Apple", "Apple iPhone sales 2025"]
    state.revision_count = 1

    state_dict = critic_node(state)
    assert state_dict["critique"] is None

    route = should_revise(state_dict)
    assert route == "done"
