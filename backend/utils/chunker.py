"""
Chunking utilities — split documents into smaller pieces for retrieval.

Used by parent-child chunking (Technique 8) and semantic chunking (Technique 9).
Basic fixed-size chunking is provided here as a foundation.
"""

from retrievers.base import Document


def fixed_size_chunk(
    documents: list[Document],
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[Document]:
    """
    Split documents into fixed-size chunks with overlap.

    Args:
        documents: List of documents to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between consecutive chunks
    """
    chunks = []
    for doc in documents:
        text = doc.content
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(Document(
                id=f"{doc.id}_chunk_{chunk_idx}",
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "parent_id": doc.id,
                    "chunk_index": chunk_idx,
                },
            ))

            start += chunk_size - overlap
            chunk_idx += 1

    return chunks
