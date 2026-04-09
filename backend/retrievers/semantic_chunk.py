"""
Technique 9: Semantic Chunking — Break at Topic Boundaries

The problem: Fixed-size chunking (e.g., every 200 characters) can split a
document in the middle of a sentence or idea, breaking context.

The fix: Split documents at natural topic boundaries by measuring the
semantic similarity between consecutive sentences. When similarity drops
sharply, that's a topic boundary — split there.

How it works:
  1. Split document into sentences
  2. Embed each sentence
  3. Compute cosine similarity between consecutive sentence pairs
  4. Where similarity drops below a threshold → topic boundary → split here
  5. Index the resulting semantic chunks and search normally

Analogy: Instead of cutting a book into equal 2-page pieces (fixed-size),
you cut it at chapter boundaries (semantic chunking). Each piece is a
coherent topic.

When semantic chunking wins:
  - Documents cover multiple topics
  - Fixed-size chunks break important context
"""

import re
import numpy as np

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from utils.embeddings import embed_texts, embed_query


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _semantic_chunk(text: str, threshold: float = 0.5) -> list[str]:
    """
    Split text into semantic chunks based on topic boundaries.

    Steps:
      1. Split into sentences
      2. Embed each sentence
      3. Compute similarity between consecutive sentences
      4. Split where similarity drops below threshold
    """
    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        return [text]

    # Embed all sentences
    embeddings = np.array(embed_texts(sentences))

    # Compute cosine similarity between consecutive sentences
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(embeddings)):
        # Cosine similarity between sentence i-1 and sentence i
        sim = float(np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        ))

        if sim < threshold:
            # Topic boundary detected — start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class SemanticChunkRetriever(BaseRetriever):
    """Chunks documents at topic boundaries using sentence similarity, then retrieves."""

    def __init__(self, threshold: float = 0.5):
        self._documents: list[Document] = []
        self._chunks: list[Document] = []
        self._chunk_embeddings: np.ndarray | None = None
        self._parent_map: dict[str, Document] = {}
        self._threshold = threshold

    def index(self, documents: list[Document]) -> None:
        self._documents = documents
        self._parent_map = {doc.id: doc for doc in documents}
        self._chunks = []

        # Semantically chunk each document
        for doc in documents:
            chunk_texts = _semantic_chunk(doc.content, self._threshold)
            for i, chunk_text in enumerate(chunk_texts):
                self._chunks.append(Document(
                    id=f"{doc.id}_semchunk_{i}",
                    content=chunk_text,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_index": i},
                ))

        # Embed all semantic chunks
        chunk_texts = [c.content for c in self._chunks]
        self._chunk_embeddings = np.array(embed_texts(chunk_texts))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._chunk_embeddings is None:
            return []

        query_vec = np.array(embed_query(query))

        # Cosine similarity
        doc_norms = np.linalg.norm(self._chunk_embeddings, axis=1, keepdims=True)
        normalized_docs = self._chunk_embeddings / doc_norms
        query_norm = np.linalg.norm(query_vec)
        normalized_query = query_vec / query_norm

        similarities = normalized_docs @ normalized_query
        top_indices = np.argsort(similarities)[::-1]

        # Deduplicate by parent, keep best chunk per parent
        seen_parents: dict[str, tuple[float, str]] = {}
        for idx in top_indices:
            chunk = self._chunks[int(idx)]
            parent_id = chunk.metadata.get("parent_id", chunk.id)
            score = float(similarities[idx])

            if parent_id not in seen_parents:
                seen_parents[parent_id] = (score, chunk.content)

            if len(seen_parents) >= top_k:
                break

        results = []
        for rank, (parent_id, (score, chunk_text)) in enumerate(seen_parents.items(), start=1):
            parent = self._parent_map[parent_id]
            chunk_preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text

            explanation = (
                f"Semantic chunk score {score:.4f} — document was split at topic boundaries. "
                f"Best matching chunk: \"{chunk_preview}\""
            )

            results.append(RetrievalResult(
                document=parent,
                score=round(score, 4),
                rank=rank,
                metadata={"technique": "semantic_chunk", "matched_chunk": chunk_text},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Semantic Chunking",
            description="Splits documents at natural topic boundaries (not fixed sizes), then searches chunks. Best when documents cover multiple topics.",
            category="post-retrieval",
        )
