from tools import rag_tools


def test_rag_search():
    results = rag_tools.search("Tesla earnings margins", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert "title" in top
    assert "text" in top
    assert "score" in top
