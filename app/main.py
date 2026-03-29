"""
FastAPI application entry point.
Settings are validated at import time — a missing OPENAI_API_KEY crashes fast
rather than surfacing as a cryptic error on the first request.
"""

import logging
import logging.config

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import ingest, query, summarize
from app.utils.config import get_settings

# ---------------------------------------------------------------------------
# Logging — configure once at startup; all module loggers inherit from root
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
})

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
app.include_router(summarize.router, prefix="/api/v1", tags=["summarize"])


# ---------------------------------------------------------------------------
# Global exception handler — normalises ALL HTTPException responses to
# {"error": "..."} instead of FastAPI's default {"detail": "..."}.
# Keeps every endpoint's error shape consistent without per-route boilerplate.
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}
