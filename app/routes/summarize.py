"""
POST /api/v1/summarize

Retrieves top-10 architecture-level chunks from the vector store and asks the
LLM to produce a structured summary (purpose, architecture, modules, tech stack).
No request body required.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.utils.dependencies import get_rag_pipeline, get_vector_store
from app.utils.models import ErrorResponse, SummarizeResponse
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "No codebase ingested"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
    },
)
async def summarize(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    vector_store: VectorStore = Depends(get_vector_store),
) -> SummarizeResponse:
    logger.info("POST /summarize — request received")

    # Fast-path guard: skip embedding + LLM call if the store is empty
    if vector_store.count() == 0:
        logger.warning("POST /summarize — vector store is empty, rejecting request")
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed yet. Please ingest a repository first.",
        )

    try:
        logger.info("POST /summarize — delegating to RAGPipeline.summarize()")
        result = pipeline.summarize()
        logger.info("POST /summarize — summary generated successfully")
        return result

    except ValueError as exc:
        # Raised by the pipeline when retrieved context is empty after filtering
        logger.warning("POST /summarize — empty context: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # Raised by the pipeline when the LLM call fails
        logger.error("POST /summarize — LLM failure: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    except Exception as exc:
        # Catch-all: log full stack trace, return a safe generic message
        logger.error("POST /summarize — unexpected error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )
