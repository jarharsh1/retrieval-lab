"""
POST /api/upload — Upload a custom dataset for retrieval testing.

Accepts a JSON file with documents in the format:
  [{ "id": "...", "content": "...", "metadata": { ... } }]

Saves it to the datasets directory so it can be used with /api/retrieve.
"""

import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from datasets.loader import DATASETS_DIR, AVAILABLE_DATASETS

router = APIRouter()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), name: str = "custom"):
    """Upload a JSON dataset file."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    content = await file.read()

    try:
        docs = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    # Validate structure
    if not isinstance(docs, list):
        raise HTTPException(status_code=400, detail="JSON must be an array of documents")

    for i, doc in enumerate(docs):
        if "id" not in doc or "content" not in doc:
            raise HTTPException(
                status_code=400,
                detail=f"Document at index {i} missing required fields: 'id' and 'content'",
            )

    # Save to datasets directory
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-").lower()
    file_path = DATASETS_DIR / f"{safe_name}.json"

    with open(file_path, "w") as f:
        json.dump(docs, f, indent=2)

    # Register the new dataset
    AVAILABLE_DATASETS[safe_name] = f"{safe_name}.json"

    return {
        "message": f"Dataset '{safe_name}' uploaded successfully",
        "document_count": len(docs),
        "name": safe_name,
    }
