"""
Technique 1: BM25 (Best Matching 25) — Sparse Keyword Retrieval

BM25 is the baseline retrieval technique. It ranks documents by how well they
match the query's keywords, adjusted for:
  - Term Frequency (TF): How often a word appears in a document
  - Inverse Document Frequency (IDF): How rare a word is across all documents
  - Document Length: Longer docs get a slight penalty (avoids bias toward verbose text)

Formula:
  score(q, d) = Σ IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * |d|/avgdl))

Where:
  k1 = 1.5 (term frequency saturation — diminishing returns for repeated words)
  b  = 0.75 (length normalization — how much to penalize long documents)

When to use BM25:
  - Query has exact keywords: "metformin 500mg", "reliability below 90%"
  - Precise term matching matters more than meaning

When BM25 fails:
  - Synonyms: "vendors" won't match "suppliers"
  - Conceptual queries: "high blood sugar" won't match "hyperglycemia"
"""

from rank_bm25 import BM25Okapi

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo


class BM25Retriever(BaseRetriever):
    """BM25 sparse keyword retriever — the baseline for all comparisons."""

    def __init__(self):
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def index(self, documents: list[Document]) -> None:
        """
        Tokenize each document and build the BM25 index.

        Tokenization here is simple: lowercase + split on whitespace.
        Production systems use stemming, stopword removal, etc.
        For a learning tool, keeping it simple makes the scores easier to understand.
        """
        self._documents = documents

        # Tokenize: "Find suppliers in China" → ["find", "suppliers", "in", "china"]
        self._tokenized_corpus = [
            doc.content.lower().split() for doc in documents
        ]

        # BM25Okapi automatically computes IDF, avgdl, and doc lengths
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Score every document against the query and return top_k results.

        Steps:
          1. Tokenize the query the same way we tokenized documents
          2. BM25 scores each document (higher = more relevant)
          3. Sort by score descending, take top_k
          4. Generate a human-readable explanation for each result
        """
        if self._bm25 is None:
            return []

        # Tokenize query the same way as documents
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Pair each document with its score, sort descending
        scored_docs = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for rank, (doc_idx, score) in enumerate(scored_docs[:top_k], start=1):
            doc = self._documents[doc_idx]
            doc_tokens = self._tokenized_corpus[doc_idx]

            # Build explanation: which query terms matched and how often
            matched_terms = []
            for token in query_tokens:
                count = doc_tokens.count(token)
                if count > 0:
                    matched_terms.append(f"'{token}' (appears {count}x)")

            if matched_terms:
                explanation = f"BM25 score {score:.2f} — matched keywords: {', '.join(matched_terms)}"
            else:
                explanation = f"BM25 score {score:.2f} — no exact keyword matches (partial/related terms contributed)"

            results.append(RetrievalResult(
                document=doc,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "bm25", "matched_terms": matched_terms},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="BM25",
            description="Sparse keyword retrieval using term frequency and inverse document frequency. Best for exact keyword matching.",
            category="sparse",
        )
