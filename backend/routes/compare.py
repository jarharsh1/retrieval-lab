"""
POST /api/compare — Run the same query through two techniques, side by side.

This is the core feature of RetrieverBench. Send:
  { query, dataset, techniques: ["bm25", "semantic"], top_k }

Get back: results from both techniques so you can see the difference.
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel

from retrievers.base import RetrievalResult
from routes.retrieve import get_retriever

router = APIRouter()


class CompareRequest(BaseModel):
    query: str
    dataset: str
    techniques: list[str]
    top_k: int = 5


class TechniqueResult(BaseModel):
    technique: str
    results: list[RetrievalResult]
    latency_ms: float
    technique_info: dict


class CompareResponse(BaseModel):
    query: str
    dataset: str
    comparisons: list[TechniqueResult]


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    comparisons = []

    for technique in req.techniques:
        retriever = get_retriever(technique, req.dataset)

        start = time.perf_counter()
        results = retriever.retrieve(req.query, req.top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        comparisons.append(
            TechniqueResult(
                technique=technique,
                results=results,
                latency_ms=round(latency_ms, 2),
                technique_info=retriever.get_info().model_dump(),
            )
        )

    return CompareResponse(
        query=req.query,
        dataset=req.dataset,
        comparisons=comparisons,
    )
