"""
Technique 8: Parent-Child Chunking — Index Small, Retrieve Big

The problem: If you index full documents, long docs dilute the relevant parts.
If you index tiny chunks, you lose surrounding context.

The fix: Index SMALL chunks (children) for precise matching, but RETURN the
FULL document (parent) so the user gets complete context.

How it works:
  1. Split each document into small chunks (children)
  2. Index the children (small = precise matching)
  3. When a child chunk matches, return the parent (full document)
  4. Deduplicate — multiple child chunks from the same parent → one result

Analogy: It's like searching a book by its index entries (small, precise)
but returning the full chapter (big, contextual) when you find a match.

When parent-child wins:
  - Documents are long and only parts are relevant
  - You need the full context around the matching passage
"""

import numpy as np

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from utils.embeddings import embed_texts, embed_query
from utils.chunker import fixed_size_chunk


class ParentChildRetriever(BaseRetriever):
    """Index small child chunks, retrieve parent (full) documents."""

    def __init__(self, chunk_size: int = 150, overlap: int = 30):
        self._parent_documents: list[Document] = []
        self._child_chunks: list[Document] = []
        self._child_embeddings: np.ndarray | None = None
        self._chunk_size = chunk_size
        self._overlap = overlap

    def index(self, documents: list[Document]) -> None:
        """
        1. Store parent documents (originals)
        2. Split into child chunks
        3. Embed the children only
        """
        self._parent_documents = documents

        # Split each parent into small child chunks
        self._child_chunks = fixed_size_chunk(
            documents,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )

        # Embed children
        child_texts = [c.content for c in self._child_chunks]
        self._child_embeddings = np.array(embed_texts(child_texts))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Search against child chunk embeddings
        2. Map matching children back to their parent documents
        3. Deduplicate parents, keep best child score per parent
        """
        if self._child_embeddings is None:
            return []

        query_vec = np.array(embed_query(query))

        # Cosine similarity against child chunks
        doc_norms = np.linalg.norm(self._child_embeddings, axis=1, keepdims=True)
        normalized_docs = self._child_embeddings / doc_norms
        query_norm = np.linalg.norm(query_vec)
        normalized_query = query_vec / query_norm

        similarities = normalized_docs @ normalized_query

        # Sort children by similarity
        top_child_indices = np.argsort(similarities)[::-1]

        # Map children back to parents, deduplicating
        parent_map = {doc.id: doc for doc in self._parent_documents}
        seen_parents: dict[str, tuple[float, str]] = {}  # parent_id → (best_score, child_content)

        for idx in top_child_indices:
            child = self._child_chunks[int(idx)]
            parent_id = child.metadata.get("parent_id", child.id)
            score = float(similarities[idx])

            if parent_id not in seen_parents:
                seen_parents[parent_id] = (score, child.content)

            if len(seen_parents) >= top_k:
                break

        results = []
        for rank, (parent_id, (score, child_text)) in enumerate(seen_parents.items(), start=1):
            parent = parent_map[parent_id]
            child_preview = child_text[:100] + "..." if len(child_text) > 100 else child_text

            explanation = (
                f"Parent-child score {score:.4f} — matched on child chunk: "
                f"\"{child_preview}\" → returned full parent document for context."
            )

            results.append(RetrievalResult(
                document=parent,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "parent_child", "matched_chunk": child_text},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Parent-Child Chunking",
            description="Indexes small chunks for precise matching but returns full parent documents for context. Best when you need both precision and complete context.",
            category="post-retrieval",
        )
