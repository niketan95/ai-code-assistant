"""
Embedding service — wraps the OpenAI Embeddings API.
Batches requests (512/call) and truncates inputs to 8191 tokens using tiktoken.
"""

import tiktoken
from openai import OpenAI

from app.utils.config import Settings

_MAX_TOKENS = 8191
_BATCH_SIZE = 512


class Embedder:
    def __init__(self, settings: Settings):
        self._model = settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        try:
            self._enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        truncated = [self._truncate(t) for t in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(truncated), _BATCH_SIZE):
            resp = self._client.embeddings.create(model=self._model, input=truncated[i : i + _BATCH_SIZE])
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _truncate(self, text: str) -> str:
        tokens = self._enc.encode(text)
        return text if len(tokens) <= _MAX_TOKENS else self._enc.decode(tokens[:_MAX_TOKENS])
