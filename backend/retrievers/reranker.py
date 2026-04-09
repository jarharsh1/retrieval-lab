"""
Technique 7: Re-Ranking — Cross-Encoder Re-Scores Initial Results

The problem: Bi-encoder (semantic search) embeds query and docs separately.
It's fast but can miss nuanced relevance because it never sees query+doc together.

The fix: First do a fast retrieval (BM25 or Semantic), then re-score the top
results using a cross-encoder that sees query AND document at the same time.

Bi-encoder (fast, used for initial retrieval):
  score = cosine_similarity(embed(query), embed(doc))  ← separate embeddings

Cross-encoder (slow, used for re-ranking):
  score = model(query + " [SEP] " + doc)  ← sees both together, much more accurate

Pipeline: Query → Semantic Search (top 20) → Cross-Encoder Re-Rank → Top 5

When re-ranking wins:
  - You need high precision (top results MUST be correct)
  - The initial retrieval returns "close but not quite" results

Trade-off: Cross-encoder is ~100x slower than bi-encoder. That's why we only
re-rank a small candidate set, not the entire corpus.
"""

from sentence_transformers import CrossEncoder

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from retrievers.semantic import SemanticRetriever

# Load cross-encoder model once
_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _cross_encoder


class RerankerRetriever(BaseRetriever):
    """Two-stage retrieval: Semantic search → Cross-encoder re-ranking."""

    def __init__(self, initial_candidates: int = 20):
        self._semantic = SemanticRetriever()
        self._initial_candidates = initial_candidates

    def index(self, documents: list[Document]) -> None:
        self._semantic.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Get initial candidates via semantic search (fast, broad)
        2. Re-score each candidate with cross-encoder (slow, precise)
        3. Re-sort by cross-encoder score, return top_k
        """
        # Stage 1: Fast initial retrieval — get more candidates than we need
        candidates = self._semantic.retrieve(query, self._initial_candidates)

        if not candidates:
            return []

        # Stage 2: Cross-encoder re-ranking
        cross_encoder = get_cross_encoder()

        # Cross-encoder expects pairs of (query, document_text)
        pairs = [(query, r.document.content) for r in candidates]
        ce_scores = cross_encoder.predict(pairs).tolist()

        # Pair candidates with their new cross-encoder scores
        scored = list(zip(candidates, ce_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (candidate, ce_score) in enumerate(scored[:top_k], start=1):
            explanation = (
                f"Re-ranked: semantic score was {candidate.score:.4f} (rank #{candidate.rank}), "
                f"cross-encoder re-scored to {ce_score:.4f} (new rank #{rank}). "
                f"Cross-encoder sees query + document together for more accurate relevance."
            )

            results.append(RetrievalResult(
                document=candidate.document,
                score=round(ce_score, 4),
                rank=rank,
                metadata={
                    "technique": "reranker",
                    "original_semantic_rank": candidate.rank,
                    "original_semantic_score": candidate.score,
                },
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Re-Ranking (Cross-Encoder)",
            description="Two-stage: fast semantic search for candidates, then cross-encoder re-scores for precision. Best when top results must be highly accurate.",
            category="post-retrieval",
        )
