"""
Central configuration — loaded once from .env via pydantic-settings.
All services receive a Settings instance through dependency injection,
so no module-level globals leak through the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # ChromaDB — stored inside data/ which is gitignored
    chroma_persist_dir: str = "./data/chroma_store"
    chroma_collection: str = "codebase"

    top_k: int = 5
    max_context_tokens: int = 6000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
