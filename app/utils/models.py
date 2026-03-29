"""
Pydantic request and response models.
Kept in one file since this project's surface area is small; split into
requests.py / responses.py if the schema grows significantly.
"""

from pydantic import BaseModel, Field


# ---------- Request ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    chat_model: str | None = None


# ---------- Response ----------

class ChunkMeta(BaseModel):
    file_path: str
    chunk_type: str   # "function" | "class" | "module" | "block"
    name: str | None
    start_line: int
    end_line: int


class IngestResponse(BaseModel):
    files_processed: int
    chunks_stored: int
    collection: str


class SourceChunk(BaseModel):
    content: str
    metadata: ChunkMeta
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str


class SummarizeResponse(BaseModel):
    summary: str
