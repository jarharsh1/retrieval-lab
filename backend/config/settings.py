from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    App configuration loaded from environment variables.
    Pydantic-settings reads from .env automatically — no hardcoded secrets.
    """

    # NVIDIA NIM — used for query rewriting, HyDE, multi-query, agentic RAG
    nvidia_nim_api_key: str = ""

    # ChromaDB connection
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # Neo4j — only needed for Technique 10 (GraphRAG)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Single instance used across the app
settings = Settings()
