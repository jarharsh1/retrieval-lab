const API_BASE = "/api";

export interface Document {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
}

export interface RetrievalResult {
  document: Document;
  score: number;
  rank: number;
  metadata: Record<string, unknown>;
  explanation: string;
}

export interface TechniqueInfo {
  name: string;
  description: string;
  category: string;
}

export interface RetrieveResponse {
  query: string;
  dataset: string;
  technique: string;
  results: RetrievalResult[];
  latency_ms: number;
  technique_info: TechniqueInfo;
}

export interface CompareResponse {
  query: string;
  dataset: string;
  comparisons: {
    technique: string;
    results: RetrievalResult[];
    latency_ms: number;
    technique_info: TechniqueInfo;
  }[];
}

export interface DatasetInfo {
  name: string;
  document_count: number;
  status: string;
}

export async function fetchRetrieve(
  query: string,
  dataset: string,
  technique: string,
  top_k: number = 5
): Promise<RetrieveResponse> {
  const res = await fetch(`${API_BASE}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, dataset, technique, top_k }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Retrieve failed");
  }
  return res.json();
}

export async function fetchCompare(
  query: string,
  dataset: string,
  techniques: string[],
  top_k: number = 5
): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, dataset, techniques, top_k }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Compare failed");
  }
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error("Failed to fetch datasets");
  const data = await res.json();
  return data.datasets;
}

export const TECHNIQUES = [
  { id: "bm25", name: "BM25", category: "sparse", phase: 1 },
  { id: "semantic", name: "Semantic Search", category: "dense", phase: 1 },
  { id: "hybrid", name: "Hybrid + RRF", category: "hybrid", phase: 1 },
  { id: "query_rewriter", name: "Query Rewriting", category: "pre-retrieval", phase: 2 },
  { id: "multi_query", name: "Multi-Query", category: "pre-retrieval", phase: 2 },
  { id: "hyde", name: "HyDE", category: "pre-retrieval", phase: 2 },
  { id: "reranker", name: "Re-Ranker", category: "post-retrieval", phase: 3 },
  { id: "parent_child", name: "Parent-Child", category: "post-retrieval", phase: 3 },
  { id: "semantic_chunk", name: "Semantic Chunking", category: "post-retrieval", phase: 3 },
  { id: "graph_rag", name: "GraphRAG", category: "advanced", phase: 4 },
  { id: "agentic", name: "Agentic RAG", category: "advanced", phase: 4 },
] as const;
