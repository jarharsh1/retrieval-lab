"""
Embedding wrapper — loads the sentence-transformers model once and reuses it.

Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - CPU-friendly (~80ms per sentence)
  - Good balance of speed vs quality for a learning/benchmarking tool
"""

from sentence_transformers import SentenceTransformer

# Load once, reuse everywhere
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into 384-dim embedding vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
