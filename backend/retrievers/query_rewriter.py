"""
Technique 4: Query Rewriting — LLM Enriches the Query Before Search

The problem: Users write vague, incomplete, or poorly worded queries.
  - "how did revolution affect farming?" → Which revolution? What kind of farming?

The fix: Ask an LLM to rewrite/expand the query BEFORE passing it to retrieval.
  - Original: "how did revolution affect farming?"
  - Rewritten: "How did the Industrial Revolution impact agricultural practices,
    crop production, and rural labor in 18th-19th century Britain?"

The rewritten query has more keywords AND more specific meaning,
so both BM25 and Semantic search perform better.

Pipeline: User Query → LLM Rewrite → Semantic Search → Results
"""

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from retrievers.semantic import SemanticRetriever
from utils.llm import llm_complete


REWRITE_PROMPT = """You are a search query optimizer. Your job is to rewrite the user's query
to be more specific, detailed, and search-friendly. Add relevant keywords and context
that would help a search engine find the best documents.

Rules:
- Keep the original intent
- Add specific terms, synonyms, and related concepts
- Expand abbreviations and vague references
- Output ONLY the rewritten query, nothing else

Original query: {query}

Rewritten query:"""


class QueryRewriterRetriever(BaseRetriever):
    """Rewrites the query using an LLM, then runs semantic search on the improved query."""

    def __init__(self):
        self._semantic = SemanticRetriever()
        self._last_rewritten_query: str = ""

    def index(self, documents: list[Document]) -> None:
        self._semantic.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Send query to LLM for rewriting
        2. Run semantic search with the rewritten query
        3. Return results with explanations showing original vs rewritten
        """
        # Step 1: LLM rewrites the query
        rewritten = llm_complete(REWRITE_PROMPT.format(query=query)).strip()
        self._last_rewritten_query = rewritten

        # Step 2: Semantic search with the improved query
        results = self._semantic.retrieve(rewritten, top_k)

        # Step 3: Update explanations to show the rewrite
        enriched_results = []
        for r in results:
            explanation = (
                f"Query rewritten: \"{query}\" → \"{rewritten}\" | "
                f"Then semantic similarity {r.score:.4f} against rewritten query"
            )
            enriched_results.append(RetrievalResult(
                document=r.document,
                score=r.score,
                rank=r.rank,
                metadata={**r.metadata, "technique": "query_rewriter", "original_query": query, "rewritten_query": rewritten},
                explanation=explanation,
            ))

        return enriched_results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Query Rewriting",
            description="LLM rewrites vague queries to be more specific before searching. Best for unclear or incomplete queries.",
            category="pre-retrieval",
        )
