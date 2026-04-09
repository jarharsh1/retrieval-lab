from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.retrieve import router as retrieve_router
from routes.compare import router as compare_router
from routes.datasets import router as datasets_router
from routes.upload import router as upload_router

app = FastAPI(
    title="RetrieverBench",
    description="Compare 11 RAG retrieval techniques side by side",
    version="0.1.0",
)

# Allow frontend (Next.js on localhost:3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(retrieve_router, prefix="/api")
app.include_router(compare_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(upload_router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}
