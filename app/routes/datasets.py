"""
app/routes/datasets.py
GET /api/datasets/{dataset_id}/file — Re-télécharge un CSV stocké sur disque.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.auth.dependencies import CurrentUser, get_current_user
from app.services.file_storage import read_upload

router = APIRouter()


@router.get("/datasets/{dataset_id}/file")
async def download_dataset_file(
    dataset_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Return the raw CSV bytes for a stored dataset."""
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id requis.")

    # Look up storage_path from DB
    try:
        import os
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import Dataset
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Dataset.storage_path).where(Dataset.id == dataset_id)
            )
            row = result.scalar_one_or_none()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur DB : {e}")

    if not row:
        raise HTTPException(status_code=404, detail="Dataset introuvable ou non stocké.")

    try:
        content = await read_upload(row)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="dataset_{dataset_id}.csv"'},
    )
