"""
Technique 5: Multi-Query Decomposition — Split Complex Queries Into Sub-Queries

The problem: Complex queries have multiple aspects that a single search can't cover.
  - "Drug interactions between blood thinners and painkillers"
    → This needs info about blood thinners AND painkillers AND their interactions

The fix: Use an LLM to decompose into sub-queries, search each separately, merge results.
  - Sub-query 1: "blood thinner anticoagulant medications warfarin"
  - Sub-query 2: "painkiller NSAID ibuprofen aspirin"
  - Sub-query 3: "drug interactions anticoagulants NSAIDs bleeding risk"

Then merge all results using RRF (same fusion as Hybrid search).

Pipeline: User Query → LLM Decompose → N Semantic Searches → RRF Merge → Results
"""

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from retrievers.semantic import SemanticRetriever
from utils.llm import llm_complete


DECOMPOSE_PROMPT = """You are a search query decomposer. Break down the following complex query
into 2-4 simpler, focused sub-queries. Each sub-query should target a different aspect
of the original question.

Rules:
- Each sub-query should be self-contained and searchable
- Cover all aspects of the original query
- Output one sub-query per line, nothing else

Original query: {query}

Sub-queries:"""


class MultiQueryRetriever(BaseRetriever):
    """Decomposes complex queries into sub-queries, searches each, merges with RRF."""

    def __init__(self, rrf_k: int = 60):
        self._semantic = SemanticRetriever()
        self._rrf_k = rrf_k
        self._documents: list[Document] = []

    def index(self, documents: list[Document]) -> None:
        self._documents = documents
        self._semantic.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. LLM decomposes query into sub-queries
        2. Run semantic search for each sub-query
        3. Merge all result lists with RRF
        """
        # Step 1: Decompose
        raw_response = llm_complete(DECOMPOSE_PROMPT.format(query=query))
        sub_queries = [q.strip().lstrip("0123456789.-) ") for q in raw_response.strip().split("\n") if q.strip()]

        if not sub_queries:
            sub_queries = [query]

        # Step 2: Search each sub-query
        all_result_lists: list[list[RetrievalResult]] = []
        for sq in sub_queries:
            results = self._semantic.retrieve(sq, top_k * 2)
            all_result_lists.append(results)

        # Step 3: RRF fusion across all sub-query result lists
        rrf_scores: dict[str, float] = {}
        for result_list in all_result_lists:
            for r in result_list:
                doc_id = r.document.id
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1 / (self._rrf_k + r.rank)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        doc_map = {doc.id: doc for doc in self._documents}

        results = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            doc = doc_map[doc_id]
            score = rrf_scores[doc_id]

            explanation = (
                f"Multi-query RRF score {score:.4f} — "
                f"Query decomposed into {len(sub_queries)} sub-queries: "
                f"{sub_queries}. Document appeared across multiple sub-query results."
            )

            results.append(RetrievalResult(
                document=doc,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "multi_query", "sub_queries": sub_queries},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Multi-Query Decomposition",
            description="LLM splits complex queries into sub-queries, searches each, and merges results with RRF. Best for queries with multiple aspects.",
            category="pre-retrieval",
        )
