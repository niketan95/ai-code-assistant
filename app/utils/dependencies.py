"""
FastAPI dependency providers using lru_cache for singleton semantics.
Override app.dependency_overrides in tests to inject mocks.
"""

from functools import lru_cache

from app.utils.config import get_settings
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore
from app.services.rag_pipeline import RAGPipeline


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(settings=get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(settings=get_settings())


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        settings=get_settings(),
    )
