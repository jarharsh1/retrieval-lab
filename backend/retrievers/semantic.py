"""
Technique 2: Dense / Semantic Search — Embedding-Based Retrieval

Instead of matching keywords, semantic search converts both the query and
documents into dense vectors (embeddings) and finds the closest ones by
cosine similarity.

How it works:
  1. Embed all documents using a sentence-transformer model → 384-dim vectors
  2. Embed the query using the same model
  3. Compute cosine similarity between query vector and every document vector
  4. Return top_k most similar documents

Why this matters:
  - "vendors" and "suppliers" have similar embeddings → semantic match!
  - "high blood sugar" and "hyperglycemia" land near each other in vector space
  - The model has learned meaning from millions of text pairs

When semantic search wins:
  - Synonyms, paraphrases, conceptual queries
  - "treatment for high blood sugar" matches docs about "hyperglycemia management"

When it fails:
  - Exact terms: "metformin 500mg" — BM25 is more precise for this
  - Rare/domain-specific terms the embedding model hasn't seen
"""

import numpy as np

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from utils.embeddings import embed_texts, embed_query


class SemanticRetriever(BaseRetriever):
    """Dense semantic search using sentence-transformer embeddings + cosine similarity."""

    def __init__(self):
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        """
        Convert all documents into embedding vectors.
        This is the expensive step — done once when the dataset is loaded.
        """
        self._documents = documents
        texts = [doc.content for doc in documents]

        # Each text → 384-dimensional vector
        self._embeddings = np.array(embed_texts(texts))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Embed the query, compute cosine similarity with all docs, return top_k.

        Cosine similarity formula:
          sim(A, B) = (A · B) / (||A|| * ||B||)

        Range: -1 (opposite) to 1 (identical). For normalized vectors, dot product = cosine similarity.
        """
        if self._embeddings is None:
            return []

        # Embed the query into the same 384-dim space
        query_vec = np.array(embed_query(query))

        # Cosine similarity = dot product of normalized vectors
        # Normalize document embeddings
        doc_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normalized_docs = self._embeddings / doc_norms

        # Normalize query
        query_norm = np.linalg.norm(query_vec)
        normalized_query = query_vec / query_norm

        # Dot product of normalized vectors = cosine similarity
        similarities = normalized_docs @ normalized_query

        # Sort by similarity descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = self._documents[idx]
            score = float(similarities[idx])

            explanation = (
                f"Cosine similarity {score:.4f} — the meaning of the query is "
                f"close to this document in embedding space (384-dim vectors)"
            )

            results.append(RetrievalResult(
                document=doc,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "semantic"},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Semantic Search",
            description="Dense retrieval using sentence-transformer embeddings and cosine similarity. Best for conceptual and synonym-based queries.",
            category="dense",
        )
