"""
GET /api/datasets — List available datasets and their metadata.
"""

from fastapi import APIRouter
from datasets.loader import list_datasets

router = APIRouter()


@router.get("/datasets")
def get_datasets():
    return {"datasets": list_datasets()}
