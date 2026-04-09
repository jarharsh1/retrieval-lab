"""
Technique 6: HyDE — Hypothetical Document Embeddings

The problem: Questions and answers live in different parts of embedding space.
  - Query: "Explain photosynthesis" (question-style)
  - Document: "Photosynthesis is a biological process..." (statement-style)
  - These embeddings may not be as close as you'd expect!

The fix: Generate a FAKE answer to the question using an LLM, then search
using that fake answer's embedding instead of the question's embedding.

Pipeline:
  User Query → LLM generates hypothetical answer → Embed the answer → Search → Results

The hypothetical answer doesn't need to be correct — it just needs to
SOUND like the kind of document we're looking for. This moves us from
"question space" to "answer space" in the embedding model.

When HyDE wins:
  - Question vs statement mismatch: "Explain photosynthesis"
  - The query style is very different from the document style

When HyDE fails:
  - The LLM generates a bad hypothetical answer
  - Factual queries where exact terms matter more than style
"""

import numpy as np

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from utils.embeddings import embed_texts, embed_query
from utils.llm import llm_complete


HYDE_PROMPT = """Answer the following question in a detailed, factual paragraph.
Write as if you are a reference document or encyclopedia entry.
Do NOT say "I don't know" — provide your best answer even if uncertain.

Question: {query}

Answer:"""


class HyDERetriever(BaseRetriever):
    """Generates a hypothetical answer, embeds it, and searches with that embedding."""

    def __init__(self):
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        self._documents = documents
        texts = [doc.content for doc in documents]
        self._embeddings = np.array(embed_texts(texts))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Generate a hypothetical answer using LLM
        2. Embed the hypothetical answer (NOT the original query)
        3. Cosine similarity search with that embedding
        """
        if self._embeddings is None:
            return []

        # Step 1: Generate hypothetical document
        hypothetical = llm_complete(HYDE_PROMPT.format(query=query)).strip()

        # Step 2: Embed the hypothetical answer
        hyde_vec = np.array(embed_query(hypothetical))

        # Step 3: Cosine similarity against all documents
        doc_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normalized_docs = self._embeddings / doc_norms
        query_norm = np.linalg.norm(hyde_vec)
        normalized_query = hyde_vec / query_norm

        similarities = normalized_docs @ normalized_query
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = self._documents[idx]
            score = float(similarities[idx])

            # Truncate hypothetical for display
            hyp_preview = hypothetical[:150] + "..." if len(hypothetical) > 150 else hypothetical

            explanation = (
                f"HyDE similarity {score:.4f} — instead of searching with the question, "
                f"an LLM generated a hypothetical answer: \"{hyp_preview}\" "
                f"and we searched with THAT embedding."
            )

            results.append(RetrievalResult(
                document=doc,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "hyde", "hypothetical_answer": hypothetical},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="HyDE (Hypothetical Document Embeddings)",
            description="Generates a fake answer via LLM, then searches with that embedding. Bridges the gap between question-style queries and statement-style documents.",
            category="pre-retrieval",
        )
