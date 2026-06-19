"""
app/services/file_storage.py
Local filesystem storage for uploaded datasets.

Files are stored at:
  {UPLOADS_ROOT}/{mission_id}/{dataset_id}/{filename}

UPLOADS_ROOT defaults to ./uploads (relative to the project root) and can be
overridden with the UPLOADS_DIR env var.  In production, point this at a
persistent volume or replace with an S3/GCS backend.
"""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles

UPLOADS_ROOT = Path(os.getenv("UPLOADS_DIR", "uploads"))


async def save_upload(content: bytes, mission_id: str, dataset_id: str, filename: str) -> str:
    """Persist `content` to disk and return the relative storage path."""
    dest_dir = UPLOADS_ROOT / mission_id / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(content)
    return str(dest_path)


async def read_upload(storage_path: str) -> bytes:
    """Read a previously stored file and return its bytes."""
    path = Path(storage_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {storage_path}")
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


def delete_upload(storage_path: str) -> bool:
    """Delete a stored file; returns True if deleted, False if already gone."""
    path = Path(storage_path)
    if path.exists():
        path.unlink()
        try:
            path.parent.rmdir()  # remove dataset dir if now empty
        except OSError:
            pass
        return True
    return False
