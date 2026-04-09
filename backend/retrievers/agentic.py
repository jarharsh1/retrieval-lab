"""
Technique 11: Agentic RAG — LLM Agent Decides Which Technique to Use

The ultimate technique: instead of the USER picking a retrieval method,
an LLM agent analyzes the query and picks the best technique automatically.

How it works:
  1. Send the query + descriptions of all techniques to an LLM
  2. LLM picks the best technique (or combination)
  3. Run the chosen technique(s)
  4. Return results with explanation of WHY the agent chose that technique

This is the "I don't know which technique to use" answer — let the AI decide.

When Agentic RAG wins:
  - You don't know your query patterns in advance
  - Queries are diverse and need different strategies
  - You want automatic optimization without manual tuning
"""

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from retrievers.bm25 import BM25Retriever
from retrievers.semantic import SemanticRetriever
from retrievers.hybrid import HybridRetriever
from utils.llm import llm_complete


AGENT_PROMPT = """You are a retrieval technique selector. Given a user query, pick the BEST
retrieval technique from the options below.

Techniques:
1. bm25 — Best for queries with exact keywords, specific terms, names, numbers
2. semantic — Best for conceptual queries, synonyms, meaning-based matching
3. hybrid — Best when you need BOTH keyword AND meaning matching

Analyze the query and respond with ONLY the technique name (bm25, semantic, or hybrid).

Query: {query}

Best technique:"""


class AgenticRetriever(BaseRetriever):
    """LLM agent selects the best retrieval technique for each query."""

    def __init__(self):
        self._techniques: dict[str, BaseRetriever] = {
            "bm25": BM25Retriever(),
            "semantic": SemanticRetriever(),
            "hybrid": HybridRetriever(),
        }
        self._documents: list[Document] = []

    def index(self, documents: list[Document]) -> None:
        self._documents = documents
        for retriever in self._techniques.values():
            retriever.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Ask LLM which technique to use
        2. Run that technique
        3. Add agent reasoning to explanations
        """
        # Step 1: Agent selects technique
        agent_response = llm_complete(AGENT_PROMPT.format(query=query)).strip().lower()

        # Parse the response — extract technique name
        chosen = "hybrid"  # default fallback
        for technique_name in self._techniques:
            if technique_name in agent_response:
                chosen = technique_name
                break

        # Step 2: Run chosen technique
        retriever = self._techniques[chosen]
        results = retriever.retrieve(query, top_k)

        # Step 3: Enrich explanations with agent reasoning
        enriched = []
        for r in results:
            explanation = (
                f"Agent chose '{chosen}' for this query. "
                f"Reasoning: {agent_response.strip()} | "
                f"Underlying result: {r.explanation}"
            )
            enriched.append(RetrievalResult(
                document=r.document,
                score=r.score,
                rank=r.rank,
                metadata={**r.metadata, "technique": "agentic", "agent_choice": chosen, "agent_reasoning": agent_response},
                explanation=explanation,
            ))

        return enriched

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="Agentic RAG",
            description="LLM agent analyzes your query and automatically selects the best retrieval technique. Best when you don't know which technique to pick.",
            category="advanced",
        )
