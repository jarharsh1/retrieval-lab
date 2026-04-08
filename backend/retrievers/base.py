"""
BaseRetriever — the abstract class every retrieval technique inherits from.

Think of this like an interface contract:
  - index()    → "Here are your documents, prepare yourself"
  - retrieve() → "Here's a query, find the best matches"
  - get_info() → "Tell me about yourself (name, description, category)"

Every technique (BM25, Semantic, Hybrid, etc.) implements these three methods.
The API routes don't care which technique they're calling — they just call
retriever.retrieve(query, top_k) and get back a uniform list of results.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel


class Document(BaseModel):
    """A single document in a dataset."""
    id: str
    content: str
    metadata: dict = {}


class RetrievalResult(BaseModel):
    """
    What comes back from a retrieval call — one per retrieved document.

    The key field is `explanation`: a plain-English string explaining WHY
    this document was retrieved. This is what makes RetrieverBench a learning
    tool, not just a benchmarking tool.

    Example explanation for BM25:
      "Score 4.32 — matched keywords: 'supplier' (TF=3, IDF=1.2), 'China' (TF=1, IDF=2.8)"

    Example explanation for Semantic:
      "Cosine similarity 0.87 — 'vendors' is semantically close to 'suppliers'"
    """
    document: Document
    score: float
    rank: int
    metadata: dict = {}
    explanation: str


class TechniqueInfo(BaseModel):
    """Metadata about a retrieval technique — shown in the UI."""
    name: str
    description: str
    category: str  # "sparse", "dense", "hybrid", "pre-retrieval", "post-retrieval", "advanced"


class BaseRetriever(ABC):
    """Abstract base class for all retrieval techniques."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """
        Ingest documents and build the internal index.
        Called once when a dataset is loaded.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Given a query, return the top_k most relevant documents.
        Each result includes a score and a plain-English explanation.
        """
        ...

    @abstractmethod
    def get_info(self) -> TechniqueInfo:
        """Return metadata about this technique (name, description, category)."""
        ...
