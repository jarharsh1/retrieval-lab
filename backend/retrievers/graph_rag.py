"""
Technique 10: GraphRAG — Knowledge Graph Retrieval via Neo4j

The problem: Some queries require multi-hop reasoning across entities.
  - "What happens if Taiwan Semiconductor shuts down?"
  - This requires: TSMC → supplies to → Apple, NVIDIA → their products → impact

Text search (BM25/Semantic) finds documents mentioning TSMC, but can't
TRAVERSE relationships between entities.

The fix: Build a knowledge graph where entities are nodes and relationships
are edges. For multi-hop queries, traverse the graph to find connected info.

Note: This technique requires Neo4j to be running. If Neo4j is not available,
it falls back to a simulated graph using document metadata relationships.

Pipeline: Query → Extract entities → Traverse graph → Retrieve connected docs → Results
"""

from retrievers.base import BaseRetriever, Document, RetrievalResult, TechniqueInfo
from utils.embeddings import embed_texts, embed_query
import numpy as np


class GraphRAGRetriever(BaseRetriever):
    """
    Graph-based retrieval using document metadata relationships.

    This implementation builds an in-memory graph from document metadata
    (region, country, type) and uses it to find connected documents.
    For a production system, this would use Neo4j with Cypher queries.
    """

    def __init__(self):
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        # Adjacency list: doc_id → list of (related_doc_id, relationship)
        self._graph: dict[str, list[tuple[str, str]]] = {}

    def index(self, documents: list[Document]) -> None:
        self._documents = documents

        # Build embeddings for semantic matching
        texts = [doc.content for doc in documents]
        self._embeddings = np.array(embed_texts(texts))

        # Build graph edges from shared metadata properties
        self._graph = {doc.id: [] for doc in documents}

        for i, doc_a in enumerate(documents):
            for j, doc_b in enumerate(documents):
                if i == j:
                    continue
                relationships = self._find_relationships(doc_a, doc_b)
                for rel in relationships:
                    self._graph[doc_a.id].append((doc_b.id, rel))

    def _find_relationships(self, doc_a: Document, doc_b: Document) -> list[str]:
        """Find shared metadata properties between two documents."""
        relationships = []
        meta_a = doc_a.metadata
        meta_b = doc_b.metadata

        if meta_a.get("country") and meta_a.get("country") == meta_b.get("country"):
            relationships.append(f"same_country:{meta_a['country']}")
        if meta_a.get("region") and meta_a.get("region") == meta_b.get("region"):
            relationships.append(f"same_region:{meta_a['region']}")
        if meta_a.get("type") and meta_a.get("type") == meta_b.get("type"):
            relationships.append(f"same_type:{meta_a['type']}")
        if meta_a.get("category") and meta_a.get("category") == meta_b.get("category"):
            relationships.append(f"same_category:{meta_a['category']}")

        return relationships

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        1. Semantic search to find seed documents
        2. Traverse graph to find connected documents (1-hop neighbors)
        3. Score: semantic_score + graph_bonus for connected docs
        """
        if self._embeddings is None:
            return []

        query_vec = np.array(embed_query(query))

        # Cosine similarity for all docs
        doc_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normalized_docs = self._embeddings / doc_norms
        query_norm = np.linalg.norm(query_vec)
        normalized_query = query_vec / query_norm
        similarities = normalized_docs @ normalized_query

        # Find seed documents (top 3 by semantic similarity)
        seed_count = min(3, len(self._documents))
        seed_indices = np.argsort(similarities)[::-1][:seed_count]

        # Collect graph neighbors of seed docs
        graph_bonus: dict[str, list[str]] = {}  # doc_id → list of relationships
        for idx in seed_indices:
            seed_id = self._documents[idx].id
            for neighbor_id, relationship in self._graph.get(seed_id, []):
                if neighbor_id not in graph_bonus:
                    graph_bonus[neighbor_id] = []
                graph_bonus[neighbor_id].append(f"{seed_id} → {relationship}")

        # Combined scoring: semantic + graph bonus
        doc_map = {doc.id: (i, doc) for i, doc in enumerate(self._documents)}
        combined_scores: list[tuple[str, float, float, list[str]]] = []

        for doc_id, (idx, doc) in doc_map.items():
            sem_score = float(similarities[idx])
            g_bonus = 0.1 * len(graph_bonus.get(doc_id, []))  # 0.1 per connection
            total = sem_score + g_bonus
            rels = graph_bonus.get(doc_id, [])
            combined_scores.append((doc_id, total, sem_score, rels))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, total, sem_score, rels) in enumerate(combined_scores[:top_k], start=1):
            _, doc = doc_map[doc_id]

            if rels:
                rel_str = "; ".join(rels[:3])
                explanation = (
                    f"GraphRAG score {total:.4f} — semantic: {sem_score:.4f} + "
                    f"graph bonus for {len(rels)} connection(s): {rel_str}"
                )
            else:
                explanation = (
                    f"GraphRAG score {total:.4f} — semantic: {sem_score:.4f}, "
                    f"no direct graph connections to seed documents"
                )

            results.append(RetrievalResult(
                document=doc,
                score=round(total, 4),
                rank=rank,
                metadata={"technique": "graph_rag", "graph_connections": rels},
                explanation=explanation,
            ))

        return results

    def get_info(self) -> TechniqueInfo:
        return TechniqueInfo(
            name="GraphRAG",
            description="Combines semantic search with knowledge graph traversal. Finds documents connected through shared entities and relationships. Best for multi-hop reasoning queries.",
            category="advanced",
        )
