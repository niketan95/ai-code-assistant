# AI Code Assistant (RAG-based)

> Ask any question about a codebase. Get answers grounded in the actual source — with file paths and line numbers.

---

## Problem

Developers waste hours reading through unfamiliar codebases just to answer basic questions:
- *Where is authentication handled?*
- *What does this service do?*
- *How does the payment flow work?*

Code search tools return raw files. LLMs hallucinate functions that don't exist. Neither solution is reliable at scale.

---

## Solution

AI Code Assistant lets you upload any codebase and ask natural language questions about it. It uses a Retrieval-Augmented Generation (RAG) pipeline that:

1. Chunks code **intelligently** — by function and class, not arbitrary line windows
2. Embeds chunks using **OpenAI embeddings** and stores them in **ChromaDB**
3. Retrieves only the most relevant code at query time
4. Passes it to an **LLM with strict grounding** — no hallucinations, always cited

Every answer includes the exact file path and line numbers it came from.

---

## Features

- **Smart code chunking** — AST-based extraction for Python (functions, classes); sliding-window with overlap for JS, Java, Go, and 10+ other languages
- **Semantic search** — cosine similarity over OpenAI `text-embedding-3-small` vectors
- **Grounded LLM answers** — context-budgeted prompts, citations in every response
- **Two ingestion modes** — upload a `.zip` or point to a local directory path
- **Re-ingest safely** — upsert semantics mean re-running never duplicates chunks
- **Production-ready API** — versioned FastAPI endpoints, Pydantic validation, dependency injection

---

## Architecture

```
User Question
     │
     ▼
[ Embedder ]  ──────────────────────────────────┐
     │                                           │
     ▼                                           ▼
[ ChromaDB ]  ── top-k chunks ──►  [ RAG Pipeline ]  ──► [ OpenAI LLM ]
                                         │
                                         ▼
                                   Answer + Sources
```

```
POST /ingest                          POST /query
     │                                     │
     ▼                                     ▼
CodeChunker (AST / sliding-window)    Embedder.embed_query()
     │                                     │
     ▼                                     ▼
Embedder.embed_texts()             VectorStore.query() → top-k chunks
     │                                     │
     ▼                                     ▼
VectorStore.upsert_chunks()        RAGPipeline.run() → LLM → response
```

---

## Demo

```bash
# 1. Ingest your codebase
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@myrepo.zip"

# {"files_processed": 42, "chunks_stored": 318, "collection": "codebase"}

# 2. Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is JWT authentication implemented?"}'

# {
#   "answer": "JWT authentication is handled in `auth/middleware.py` lines 34-78.
#              The token is verified using the `verify_token()` function [auth/middleware.py:34-52]
#              and the decoded payload is attached to the request context [auth/middleware.py:53-78].",
#   "sources": [
#     { "file_path": "auth/middleware.py", "start_line": 34, "end_line": 78, "score": 0.94 },
#     { "file_path": "auth/utils.py",      "start_line": 12, "end_line": 29, "score": 0.87 }
#   ],
#   "model_used": "gpt-4o-mini"
# }
```

> Add screenshots or a screen recording here.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Chunking | Python `ast` module + regex sliding-window |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| LLM | OpenAI `gpt-4o-mini` (swappable per request) |
| Validation | Pydantic v2 + pydantic-settings |
| Token counting | tiktoken |

---

## Project Structure

```
ai-code-assistant/
├── app/
│   ├── main.py                  # FastAPI app, CORS, router registration
│   ├── routes/
│   │   ├── ingest.py            # POST /api/v1/ingest
│   │   └── query.py             # POST /api/v1/query
│   ├── services/
│   │   ├── chunker.py           # AST-based + sliding-window code chunker
│   │   ├── embedder.py          # Batched OpenAI embeddings with token truncation
│   │   ├── vector_store.py      # ChromaDB upsert + similarity retrieval
│   │   └── rag_pipeline.py      # Retrieval → prompt → LLM → response
│   └── utils/
│       ├── config.py            # Environment config via pydantic-settings
│       ├── dependencies.py      # FastAPI DI providers (lru_cache singletons)
│       └── models.py            # Request / response Pydantic schemas
├── data/                        # ChromaDB persistence — gitignored
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/ai-code-assistant.git
cd ai-code-assistant
```

**2. Install dependencies**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**
```bash
cp .env.example .env
# Open .env and set:  OPENAI_API_KEY=sk-...
```

**4. Run the server**
```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ingest` | Upload a `.zip` or pass a server-side `path` to index a codebase |
| `POST` | `/api/v1/query` | Ask a natural language question; returns answer + cited sources |
| `GET` | `/health` | Liveness probe |

### Ingest — form fields

| Field | Type | Description |
|---|---|---|
| `file` | File (optional) | `.zip` archive of the codebase |
| `path` | string (optional) | Absolute server-side directory path |

### Query — JSON body

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question |
| `top_k` | int | 5 | Number of chunks to retrieve (1–20) |
| `chat_model` | string | `gpt-4o-mini` | Override the LLM for this request |

---

## Supported Languages

Python, JavaScript, TypeScript, JSX/TSX, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin

---

## Future Improvements

- **Web UI** — drag-and-drop codebase upload, chat interface with syntax-highlighted source previews
- **Multi-repo support** — query across multiple indexed codebases simultaneously
- **GitHub integration** — ingest directly from a GitHub URL via the API
- **Incremental re-indexing** — detect changed files via git diff and re-embed only what changed
- **Code visualization** — dependency graphs and call trees alongside answers
- **Auth + multi-tenancy** — per-user collections for SaaS deployment
- **FAISS backend** — optional high-performance in-memory vector store for large repos

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add X"`
4. Push and open a PR

---

## License

MIT

## 🚀 Live Demo

Frontend (UI): https://aicode-assist.streamlit.app/  
Backend API: https://ai-code-assistant-yieh.onrender.com  

## What it does
AI-powered system that allows developers to upload a codebase and ask questions about it using retrieval-augmented generation (RAG).

## Why this matters
Understanding large codebases is time-consuming. This tool reduces onboarding time by enabling semantic search and contextual explanations.
