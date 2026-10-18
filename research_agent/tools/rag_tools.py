"""
Semantic RAG (Retrieval-Augmented Generation) Index.

Uses scikit-learn's TfidfVectorizer with sublinear TF scaling, unigram + bigram
n-gram feature extraction, and cosine similarity over vector embeddings for
context-aware semantic document retrieval across the internal knowledge base.
"""

from __future__ import annotations

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

import math
import re
from collections import Counter
from tools.knowledge_base import DOCUMENTS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _SemanticRagIndex:
    """Hybrid Semantic Vector Index over knowledge_base.DOCUMENTS."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_texts = [
            f"{d['title']} {' '.join(d.get('tags', []))} {d['text']}"
            for d in documents
        ]

        if _SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
            )
            self.doc_embeddings = self.vectorizer.fit_transform(self.doc_texts)
        else:
            self.doc_tokens = [_tokenize(t) for t in self.doc_texts]
            self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
            df: Counter = Counter()
            for tf in self.doc_term_freqs:
                for term in tf:
                    df[term] += 1
            self.df = df
            self.n_docs = len(documents)
            self.doc_vectors = [
                self._tfidf_vector(tf, len(tokens))
                for tf, tokens in zip(self.doc_term_freqs, self.doc_tokens)
            ]
            self.doc_norms = [
                math.sqrt(sum(v * v for v in vec.values())) or 1.0
                for vec in self.doc_vectors
            ]

    def _idf(self, term: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df.get(term, 0))) + 1.0

    def _tfidf_vector(self, term_freqs: Counter, doc_len: int) -> dict[str, float]:
        if doc_len == 0:
            return {}
        return {
            term: (count / doc_len) * self._idf(term)
            for term, count in term_freqs.items()
        }

    def query(self, text: str, top_k: int = 3) -> list[tuple[dict, float]]:
        if not text.strip():
            return []

        if _SKLEARN_AVAILABLE:
            q_vector = self.vectorizer.transform([text])
            sim_scores = cosine_similarity(q_vector, self.doc_embeddings)[0]
            ranked_indices = np.argsort(sim_scores)[::-1]

            results = []
            for idx in ranked_indices:
                score = float(sim_scores[idx])
                if score > 0:
                    results.append((self.documents[idx], score))
                if len(results) >= top_k:
                    break
            return results
        else:
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
                dot = sum(
                    q_vector.get(term, 0.0) * weight
                    for term, weight in doc_vector.items()
                )
                similarity = dot / (q_norm * doc_norm)
                if similarity > 0:
                    scores.append((doc, similarity))

            scores.sort(key=lambda pair: pair[1], reverse=True)
            return scores[:top_k]


_INDEX = _SemanticRagIndex(DOCUMENTS)


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k most relevant knowledge-base documents using semantic vector search."""
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
