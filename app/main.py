"""
FastAPI application entry point.
Settings are validated at import time — a missing OPENAI_API_KEY crashes fast
rather than surfacing as a cryptic error on the first request.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ingest, query
from app.utils.config import get_settings

get_settings()  # validate env vars early

app = FastAPI(
    title="AI Code Assistant",
    description="RAG-based AI assistant for codebases.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}
