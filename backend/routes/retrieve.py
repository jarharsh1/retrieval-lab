"""
POST /api/retrieve — Run a single retrieval query.

This is the main endpoint. The frontend sends:
  { query, dataset, technique, top_k }

And gets back: retrieved chunks with scores, explanations, and latency.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from datasets.loader import load_dataset
from retrievers.base import RetrievalResult
from retrievers.bm25 import BM25Retriever
from retrievers.semantic import SemanticRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.query_rewriter import QueryRewriterRetriever
from retrievers.multi_query import MultiQueryRetriever
from retrievers.hyde import HyDERetriever
from retrievers.reranker import RerankerRetriever
from retrievers.parent_child import ParentChildRetriever
from retrievers.semantic_chunk import SemanticChunkRetriever
from retrievers.graph_rag import GraphRAGRetriever
from retrievers.agentic import AgenticRetriever

# Technique registry — maps API name to retriever class
TECHNIQUE_REGISTRY: dict = {
    "bm25": BM25Retriever,
    "semantic": SemanticRetriever,
    "hybrid": HybridRetriever,
    "query_rewriter": QueryRewriterRetriever,
    "multi_query": MultiQueryRetriever,
    "hyde": HyDERetriever,
    "reranker": RerankerRetriever,
    "parent_child": ParentChildRetriever,
    "semantic_chunk": SemanticChunkRetriever,
    "graph_rag": GraphRAGRetriever,
    "agentic": AgenticRetriever,
}

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    dataset: str
    technique: str
    top_k: int = 5


class RetrieveResponse(BaseModel):
    query: str
    dataset: str
    technique: str
    results: list[RetrievalResult]
    latency_ms: float
    technique_info: dict


def get_retriever(technique: str, dataset: str):
    """Instantiate a retriever, load the dataset, and index it."""
    if technique not in TECHNIQUE_REGISTRY:
        available = list(TECHNIQUE_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown technique: '{technique}'. Available: {available}",
        )

    documents = load_dataset(dataset)
    retriever_class = TECHNIQUE_REGISTRY[technique]
    retriever = retriever_class()
    retriever.index(documents)
    return retriever


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    retriever = get_retriever(req.technique, req.dataset)

    start = time.perf_counter()
    results = retriever.retrieve(req.query, req.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    return RetrieveResponse(
        query=req.query,
        dataset=req.dataset,
        technique=req.technique,
        results=results,
        latency_ms=round(latency_ms, 2),
        technique_info=retriever.get_info().model_dump(),
    )
