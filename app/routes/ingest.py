"""
POST /api/v1/ingest

Accepts either:
  - multipart `file` field: a .zip archive of the codebase
  - multipart `path` field: an absolute server-side directory path

Chunks code intelligently (AST for Python, sliding-window for others),
generates embeddings, and upserts into ChromaDB.
"""

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.utils.dependencies import get_embedder, get_vector_store
from app.utils.models import IngestResponse
from app.services.chunker import CodeChunker
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore

router = APIRouter()
_chunker = CodeChunker()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile | None = File(default=None, description=".zip archive of the codebase"),
    path: str | None = Form(default=None, description="Absolute server-side directory path"),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    if file is None and path is None:
        raise HTTPException(status_code=422, detail="Provide a `file` upload or a `path` field.")

    tmp_dir: Path | None = None
    try:
        if file is not None:
            tmp_dir, source_dir = await _extract_zip(file)
        else:
            source_dir = Path(path)  # type: ignore[arg-type]
            if not source_dir.is_dir():
                raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

        chunks = _chunker.chunk_directory(source_dir)
        if not chunks:
            raise HTTPException(status_code=400, detail="No supported source files found.")

        embeddings = embedder.embed_texts([c.content for c in chunks])
        vector_store.upsert_chunks(chunks, embeddings)

        return IngestResponse(
            files_processed=len({c.file_path for c in chunks}),
            chunks_stored=len(chunks),
            collection=vector_store._collection.name,
        )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


async def _extract_zip(upload: UploadFile) -> tuple[Path, Path]:
    if not (upload.filename or "").endswith(".zip"):
        raise HTTPException(status_code=422, detail="Only .zip files are accepted.")
    data = await upload.read()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise HTTPException(status_code=422, detail="Not a valid ZIP archive.")
    tmp = Path(tempfile.mkdtemp(prefix="coderag_"))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = tmp / Path(member.filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member.filename))
    return tmp, tmp
