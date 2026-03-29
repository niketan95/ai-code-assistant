"""
POST /api/v1/summarize

Retrieves top-10 architecture-level chunks from the vector store and asks the
LLM to produce a structured summary covering purpose, architecture, key modules,
and technologies. No request body required.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.utils.dependencies import get_rag_pipeline, get_vector_store
from app.utils.models import SummarizeResponse
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStore

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    vector_store: VectorStore = Depends(get_vector_store),
) -> SummarizeResponse:
    if vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No codebase ingested yet. Call /ingest first.")
    return pipeline.summarize()
