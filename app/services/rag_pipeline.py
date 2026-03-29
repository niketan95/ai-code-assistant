"""
RAG pipeline: embed question → retrieve chunks → build prompt → call LLM.

Context budget is enforced with tiktoken so the prompt never silently exceeds
the model's context window. Retrieved sources are returned alongside the answer
so the caller can render citations.
"""

import tiktoken
from openai import OpenAI

from app.utils.config import Settings
from app.utils.models import QueryResponse, SourceChunk
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore

_SYSTEM_PROMPT = (
    "You are an expert software engineer assistant. "
    "Answer the user's question using ONLY the code snippets in the context below. "
    "Cite each claim with its file path and line range, e.g. `[auth.py:12-45]`. "
    "If the answer is not in the context, say so — do not hallucinate."
)


class RAGPipeline:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, settings: Settings):
        self._embedder = embedder
        self._store = vector_store
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)
        try:
            self._enc = tiktoken.encoding_for_model(settings.chat_model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def run(self, question: str, top_k: int | None = None, chat_model: str | None = None) -> QueryResponse:
        k = top_k or self._settings.top_k
        model = chat_model or self._settings.chat_model

        query_vec = self._embedder.embed_query(question)
        candidates: list[SourceChunk] = self._store.query(query_vec, top_k=k)
        context, sources = self._build_context(candidates)

        response = self._client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"### Context\n{context}\n\n### Question\n{question}"},
            ],
        )
        return QueryResponse(
            answer=response.choices[0].message.content or "",
            sources=sources,
            model_used=model,
        )

    def _build_context(self, chunks: list[SourceChunk]) -> tuple[str, list[SourceChunk]]:
        budget = self._settings.max_context_tokens
        parts, used, tokens = [], [], 0
        for chunk in chunks:
            snippet = (
                f"# {chunk.metadata.file_path} "
                f"(lines {chunk.metadata.start_line}-{chunk.metadata.end_line})\n"
                f"{chunk.content}\n"
            )
            t = len(self._enc.encode(snippet))
            if tokens + t > budget:
                break
            parts.append(snippet)
            used.append(chunk)
            tokens += t
        return "\n---\n".join(parts), used
