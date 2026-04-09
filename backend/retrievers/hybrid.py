"""
Technique 3: Hybrid Search + Reciprocal Rank Fusion (RRF)

The idea: BM25 is great at exact keywords, Semantic is great at meaning.
Why not run BOTH and merge the results?

Reciprocal Rank Fusion (RRF) formula:
  RRF_score(doc) = Σ  1 / (k + rank_in_list_i)

Where:
  k = 60 (constant to prevent top-ranked docs from dominating too much)
  rank_in_list_i = the doc's rank in each retriever's result list

Example:
  Doc A is rank 1 in BM25, rank 5 in Semantic:
    RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

  Doc B is rank 3 in BM25, rank 2 in Semantic:
    RRF = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

  Doc B wins! It was consistently good in both, even though Doc A was #1 in BM25.

When hybrid wins:
  - "Find vendors in China" — BM25 catches "China", Semantic catches "vendors" ≈ "suppliers"
  - Queries that need both exact matching AND conceptual understanding
"""

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from retrievers.bm25 import BM25Retriever
from retrievers.semantic import SemanticRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid search combining BM25 + Semantic with Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        # RRF constant — higher k = more equal weighting between ranks
        self._rrf_k = rrf_k
        self._bm25 = BM25Retriever()
        self._semantic = SemanticRetriever()
        self._documents: list[Document] = []

    def index(self, documents: list[Document]) -> None:
        """Index documents in BOTH retrievers."""
        self._documents = documents
        self._bm25.index(documents)
        self._semantic.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Run BM25 and Semantic search, then fuse results with RRF.

        Steps:
          1. Get top results from BM25
          2. Get top results from Semantic
          3. For each document, compute RRF score from its ranks in both lists
          4. Sort by RRF score, return top_k
        """
        # Fetch more candidates than top_k from each — RRF works better with larger pools
        pool_size = min(top_k * 3, len(self._documents))

        bm25_results = self._bm25.retrieve(query, pool_size)
        semantic_results = self._semantic.retrieve(query, pool_size)

        # Build rank maps: doc_id → rank in each list
        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}
        for r in bm25_results:
            bm25_ranks[r.document.id] = r.rank
            bm25_scores[r.document.id] = r.score

        semantic_ranks: dict[str, int] = {}
        semantic_scores: dict[str, float] = {}
        for r in semantic_results:
            semantic_ranks[r.document.id] = r.rank
            semantic_scores[r.document.id] = r.score

        # Collect all unique document IDs from both result sets
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

        # Compute RRF score for each document
        # If a doc didn't appear in one list, use a large rank (pool_size + 1)
        rrf_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id, pool_size + 1)
            sem_rank = semantic_ranks.get(doc_id, pool_size + 1)

            rrf_score = (1 / (self._rrf_k + bm25_rank)) + (1 / (self._rrf_k + sem_rank))
            rrf_scores[doc_id] = rrf_score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Build a doc_id → Document lookup
        doc_map = {doc.id: doc for doc in self._documents}

        results = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            doc = doc_map[doc_id]
            score = rrf_scores[doc_id]

            bm25_r = bm25_ranks.get(doc_id, "not in top")
            sem_r = semantic_ranks.get(doc_id, "not in top")
            bm25_s = f"{bm25_scores.get(doc_id, 0):.2f}"
            sem_s = f"{semantic_scores.get(doc_id, 0):.4f}"

            explanation = (
                f"RRF score {score:.4f} — "
                f"BM25 rank #{bm25_r} (score {bm25_s}), "
                f"Semantic rank #{sem_r} (similarity {sem_s}). "
                f"Documents ranked highly by BOTH techniques score best."
            )

            results.append(RetrievalResult(
                document=doc,
                score=round(score, 4),
                rank=rank,
                metadata={
                    "technique": "hybrid",
                    "bm25_rank": bm25_r,
                    "semantic_rank": sem_r,
                },
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Hybrid Search + RRF",
            description="Combines BM25 (keywords) and Semantic (meaning) using Reciprocal Rank Fusion. Best when you need both exact terms and conceptual matching.",
            category="hybrid",
        )
