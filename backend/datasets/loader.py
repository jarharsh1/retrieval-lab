"""
Dataset loader — reads JSON dataset files and converts them to Document objects.

Each dataset is a JSON array of objects with at least:
  { "id": "...", "content": "...", "metadata": { ... } }
"""

import json
from pathlib import Path
from retrievers.base import Document

# All dataset files live in this directory
DATASETS_DIR = Path(__file__).parent

# Supported datasets and their filenames
AVAILABLE_DATASETS = {
    "supply_chain": "supply_chain.json",
    "healthcare": "healthcare.json",
    "wikipedia": "wikipedia.json",
}


def load_dataset(name: str) -> list[Document]:
    """Load a dataset by name and return a list of Document objects."""
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(AVAILABLE_DATASETS.keys())}")

    file_path = DATASETS_DIR / AVAILABLE_DATASETS[name]
    with open(file_path, "r") as f:
        raw_docs = json.load(f)

    return [
        Document(
            id=doc["id"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        for doc in raw_docs
    ]


def list_datasets() -> list[dict]:
    """Return metadata about all available datasets."""
    datasets = []
    for name, filename in AVAILABLE_DATASETS.items():
        file_path = DATASETS_DIR / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                docs = json.load(f)
            datasets.append({
                "name": name,
                "document_count": len(docs),
                "status": "available",
            })
        else:
            datasets.append({
                "name": name,
                "document_count": 0,
                "status": "missing",
            })
    return datasets
