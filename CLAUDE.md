# CLAUDE.md — RetrieverBench Project Context

## What This Project Is

RetrieverBench is an interactive playground for learning, testing, and comparing document retrieval techniques used in modern RAG (Retrieval-Augmented Generation) systems. Users pick a dataset, pick a retrieval technique, run a query, and see exactly what got retrieved, why, and how fast. The core feature is side-by-side comparison — running the same query through two different techniques to see why one finds better results than the other.

This is both a learning tool (for engineers preparing for ML/AI interviews) and a benchmarking tool (for comparing retrieval strategies on real data).

## Who I Am

I'm Harsh. I work in healthcare analytics and AI at Aster Healthcare (UAE). I communicate frequently in Hinglish (Hindi-English mix) and prefer a direct, results-oriented style. I like incremental, hands-on guidance over large batches of instructions. I prefer building things myself and understanding the code rather than copy-pasting generated code. When explaining concepts, use simple analogies and break things down step by step.

## Tech Stack

- **Backend:** FastAPI (Python 3.11+) running in Docker
- **Vector Database:** ChromaDB (lightweight, local, no external dependencies)
- **Graph Database:** Neo4j (via Docker) for the GraphRAG module
- **Embedding Model:** `all-MiniLM-L6-v2` via sentence-transformers (CPU-friendly, 384-dim vectors)
- **Re-Ranking Model:** `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Sparse Retrieval:** `rank-bm25` (pure Python BM25)
- **LLM:** NVIDIA NIM (Llama 3.1 70B Instruct) — used for query rewriting, HyDE, multi-query decomposition, agentic RAG, and answer generation. API key stored in `.env` as `NVIDIA_NIM_API_KEY`
- **Frontend:** Next.js 14 + Tailwind CSS + Shadcn/ui, deployed on Vercel
- **Containerization:** Docker Compose (FastAPI backend + ChromaDB + Neo4j)

## Project Structure

```
RetrieverBench/
├── CLAUDE.md                       # You're reading this
├── README.md                       # Project documentation
├── docker-compose.yml              # FastAPI + ChromaDB + Neo4j
├── .env.example                    # API keys template
├── .env                            # Actual keys (gitignored)
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                     # FastAPI app entry point
│   │
│   ├── config/
│   │   └── settings.py             # Pydantic settings, env config
│   │
│   ├── datasets/
│   │   ├── loader.py               # Dataset loading + preprocessing
│   │   ├── supply_chain.json       # Supply chain docs
│   │   ├── healthcare.json         # Medical docs
│   │   └── wikipedia.json          # Wikipedia subset
│   │
│   ├── retrievers/                 # CORE — each technique = one file
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseRetriever abstract class
│   │   ├── bm25.py                 # Technique 1
│   │   ├── semantic.py             # Technique 2
│   │   ├── hybrid.py               # Technique 3
│   │   ├── query_rewriter.py       # Technique 4
│   │   ├── multi_query.py          # Technique 5
│   │   ├── hyde.py                 # Technique 6
│   │   ├── reranker.py             # Technique 7
│   │   ├── parent_child.py         # Technique 8
│   │   ├── semantic_chunk.py       # Technique 9
│   │   ├── graph_rag.py            # Technique 10
│   │   └── agentic.py              # Technique 11
│   │
│   ├── routes/
│   │   ├── retrieve.py             # POST /api/retrieve
│   │   ├── compare.py              # POST /api/compare
│   │   ├── datasets.py             # GET /api/datasets
│   │   └── upload.py               # POST /api/upload
│   │
│   └── utils/
│       ├── embeddings.py           # Sentence-transformers wrapper
│       ├── llm.py                  # NVIDIA NIM client
│       ├── metrics.py              # Latency tracking, scoring
│       └── chunker.py              # Chunking utilities
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Main playground page
│   │   └── globals.css
│   ├── components/
│   │   ├── DatasetSelector.tsx
│   │   ├── TechniquePicker.tsx
│   │   ├── QueryInput.tsx
│   │   ├── ResultsPanel.tsx
│   │   ├── ComparisonView.tsx
│   │   ├── ScoreBreakdown.tsx
│   │   ├── LatencyBar.tsx
│   │   └── TechniqueInfo.tsx
│   └── lib/
│       └── api.ts                  # Backend API client
│
└── notebooks/
    ├── 01_bm25_math.ipynb
    ├── 02_embeddings.ipynb
    ├── 03_rrf_fusion.ipynb
    └── 04_chunking_strategies.ipynb
```

## Build Plan — Technique by Technique

We are building this incrementally. Each technique gets built, tested, and committed before moving to the next. The order matters because each technique builds on the previous one.

### Phase 1 — Foundation
1. **BM25** (sparse retrieval) — baseline everything gets compared against
2. **Dense/Semantic Search** — embeddings + cosine similarity via ChromaDB
3. **Hybrid Search + RRF Fusion** — combine BM25 + semantic, merge with RRF

### Phase 2 — Pre-Retrieval Enhancements
4. **Query Rewriting** — LLM enriches query before search
5. **Multi-Query Decomposition** — split complex queries into sub-queries
6. **HyDE** — generate fake answer, search with that embedding

### Phase 3 — Post-Retrieval & Chunking
7. **Re-Ranking** — cross-encoder re-scores results
8. **Parent-Child Chunking** — index small, retrieve big
9. **Semantic Chunking** — break at topic boundaries

### Phase 4 — Advanced
10. **GraphRAG** — Neo4j knowledge graph retrieval
11. **Agentic RAG** — LLM agent decides which technique to use

### Current Status
- [x] README.md created
- [x] CLAUDE.md created
- [ ] Project scaffolding (docker-compose, requirements, base retriever)
- [ ] Technique 1: BM25 — NEXT UP

## Coding Conventions

### Python (Backend)
- Every retriever module must inherit from `BaseRetriever` in `base.py`
- Every retriever must implement: `index(documents)`, `retrieve(query, top_k)`, and `get_info()` methods
- The `retrieve()` method must return a list of `RetrievalResult` objects containing: `document`, `score`, `rank`, `metadata`, and `explanation` (a plain-English string explaining WHY this doc was retrieved)
- Add detailed inline comments explaining the theory and math behind each step — the code IS the tutorial
- Use type hints everywhere
- Use Pydantic models for request/response schemas
- f-strings for formatting, no .format() or % formatting
- Environment variables via pydantic-settings, never hardcoded

### TypeScript (Frontend)
- Functional components with hooks, no class components
- Tailwind CSS for all styling, Shadcn/ui for complex components
- Keep components small and focused — one component per file
- Use TypeScript strict mode

### Git
- One commit per technique: "feat: add BM25 sparse retrieval"
- Descriptive commit messages following conventional commits
- No secrets in commits — .env is gitignored

### Docker
- Backend Dockerfile uses multi-stage build
- docker-compose.yml defines three services: backend, chromadb, neo4j
- All services on same Docker network for inter-service communication
- Neo4j only needed from Technique 10 (GraphRAG) onwards

## Datasets

Three built-in datasets, each highlighting different retrieval strengths:

### Supply Chain
Documents about suppliers, components, products, warehouses, and retailers. Sample queries:
- "Which suppliers have reliability below 90%?" (BM25 shines — exact property matching)
- "What happens if Taiwan Semiconductor shuts down?" (GraphRAG needed — multi-hop)
- "Find vendors in China" (Semantic wins — "vendors" ≈ "suppliers")

### Healthcare
Medical documents about drugs, prescriptions, treatment protocols. Sample queries:
- "Side effects of metformin 500mg" (BM25 wins — exact drug name + dosage)
- "Treatment options for high blood sugar" (Semantic wins — "high blood sugar" ≈ "hyperglycemia")
- "Drug interactions between blood thinners and painkillers" (Multi-query needed — two aspects)

### Wikipedia (General Knowledge)
Curated articles across science, history, tech, geography. Sample queries:
- "How did the industrial revolution affect farming?" (Query rewriting helps — vague query)
- "Tall art deco buildings built in New York in the 1930s" (Multi-query — multiple constraints)
- "Explain photosynthesis" (HyDE helps — question vs statement style mismatch)

## API Design

### POST /api/retrieve
```json
{
  "query": "Which suppliers have reliability below 90%?",
  "dataset": "supply_chain",
  "technique": "bm25",
  "top_k": 5
}
```
Response includes: retrieved chunks with scores, latency_ms, technique_info, and per-chunk explanations.

### POST /api/compare
```json
{
  "query": "Find vendors in China",
  "dataset": "supply_chain",
  "techniques": ["bm25", "semantic"],
  "top_k": 5
}
```
Response includes results from both techniques side by side for comparison.

## Important Context From Past Work

- I built ChainMind (a GraphRAG supply chain intelligence engine) using Neo4j AuraDB + FastAPI + Next.js + NVIDIA NIM (Llama 3.1 70B). The schema pruning, Cypher chain, and chatbot pipeline from ChainMind will directly inform the GraphRAG module here.
- I'm comfortable with Neo4j, Cypher queries, FastAPI, Next.js, Tailwind, and Docker.
- I have an NVIDIA NIM API key already set up.
- I understand the retrieval techniques conceptually (went through a detailed quiz-based learning session) — now I want to implement each one from scratch.

## How To Help Me

- Explain what each piece of code does and WHY before writing it
- Keep changes incremental — one file at a time, test before moving on
- Add heavy inline comments explaining theory and math
- When I'm confused, use simple analogies (you've seen my learning style in past conversations)
- Don't generate massive boilerplate — build up step by step
- Always test with a sample query after implementing each technique
- After each technique is working, suggest a commit message and we'll push

## Commands Reference

```bash
# Start all services
docker compose up -d

# Start backend only (dev mode)
cd backend && uvicorn main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev

# Run a quick test query against BM25
curl -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "suppliers in China", "dataset": "supply_chain", "technique": "bm25", "top_k": 5}'

# Check Docker services
docker compose ps

# View backend logs
docker compose logs -f backend
```
