"""
POST /api/v1/query

Embeds the question, retrieves top-k code chunks, and returns an LLM answer
grounded in the indexed codebase with source citations.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.utils.dependencies import get_rag_pipeline, get_vector_store
from app.utils.models import QueryRequest, QueryResponse
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStore

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    vector_store: VectorStore = Depends(get_vector_store),
) -> QueryResponse:
    if vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No codebase ingested yet. Call /ingest first.")
    return pipeline.run(question=request.question, top_k=request.top_k, chat_model=request.chat_model)
