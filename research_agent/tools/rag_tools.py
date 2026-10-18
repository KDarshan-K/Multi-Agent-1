"""
RAG (retrieval-augmented generation) tool.

The retrieval mechanism here is REAL — pure-Python TF-IDF + cosine
similarity over tools/knowledge_base.py's mock document corpus. This is
not a stub; it actually scores and ranks documents by relevance to a
query. What's mocked is only the underlying *content* being retrieved
(see knowledge_base.py's docstring).

WHY TF-IDF INSTEAD OF EMBEDDINGS: a real production RAG system would
usually use a vector DB (Chroma/FAISS/pgvector) with embeddings from an
API (OpenAI, Voyage, Anthropic's embedding partners, etc.) for semantic
similarity. That needs network access this sandbox doesn't have. TF-IDF is
a legitimate, fully-local alternative that works well for keyword-heavy
financial text (ticker symbols, terms like "earnings", "guidance",
"margin") even though it won't catch pure paraphrases the way embeddings
would.

TO GO LIVE with real embeddings:
    - pip install chromadb (or use FAISS directly)
    - embed each doc in knowledge_base.py once at index-build time using
      an embedding API
    - replace `_tfidf_search()` below with a vector similarity query
    - keep the same `search(query, top_k)` function signature so nothing
      upstream (mcp_server, rag_agent) needs to change
"""

from __future__ import annotations
import math
import re
from collections import Counter

from tools.knowledge_base import DOCUMENTS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _TfidfIndex:
    """Built once at import time over knowledge_base.DOCUMENTS."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_tokens = [_tokenize(d["title"] + " " + d["text"]) for d in documents]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        # document frequency: how many docs contain each term
        df: Counter = Counter()
        for tf in self.doc_term_freqs:
            for term in tf:
                df[term] += 1
        self.df = df
        self.n_docs = len(documents)

        # precompute TF-IDF vectors for each doc
        self.doc_vectors = [
            self._tfidf_vector(tf, len(tokens))
            for tf, tokens in zip(self.doc_term_freqs, self.doc_tokens)
        ]
        self.doc_norms = [
            math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for vec in self.doc_vectors
        ]

    def _idf(self, term: str) -> float:
        # +1 smoothing to avoid division by zero / log(0)
        return math.log((1 + self.n_docs) / (1 + self.df.get(term, 0))) + 1.0

    def _tfidf_vector(self, term_freqs: Counter, doc_len: int) -> dict[str, float]:
        if doc_len == 0:
            return {}
        return {
            term: (count / doc_len) * self._idf(term)
            for term, count in term_freqs.items()
        }

    def query(self, text: str, top_k: int = 3) -> list[tuple[dict, float]]:
        q_tokens = _tokenize(text)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_vector = self._tfidf_vector(q_tf, len(q_tokens))
        q_norm = math.sqrt(sum(v * v for v in q_vector.values())) or 1.0

        scores = []
        for doc, doc_vector, doc_norm in zip(
            self.documents, self.doc_vectors, self.doc_norms
        ):
            # cosine similarity = dot product / (norm_a * norm_b)
            dot = sum(
                q_vector.get(term, 0.0) * weight
                for term, weight in doc_vector.items()
            )
            similarity = dot / (q_norm * doc_norm)
            if similarity > 0:
                scores.append((doc, similarity))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]


_INDEX = _TfidfIndex(DOCUMENTS)


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k most relevant knowledge-base docs for `query`."""
    results = _INDEX.query(query, top_k=top_k)
    return [
        {
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "tags": doc.get("tags", []),
            "score": round(score, 4),
        }
        for doc, score in results
    ]
