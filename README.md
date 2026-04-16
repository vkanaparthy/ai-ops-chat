# ai-ops-chat

Agentic **Root Cause Analysis (RCA)** over structured application logs: watch a folder, parse pipe-delimited lines, embed English log text with **Azure OpenAI**, store vectors in **ChromaDB**, and answer questions through a **Strands** agent on **Amazon Bedrock** via a **FastAPI** `POST /ai/chat` endpoint.

## Prerequisites

- Python 3.10+
- [Node.js](https://nodejs.org/) (includes `npm`) — only needed for the `frontend/` Vite UI
- An **Azure OpenAI** resource with an embedding deployment (e.g. `text-embedding-ada-002`) and a valid API key
- AWS credentials with permission to invoke your Bedrock model (`bedrock:InvokeModel`), and the model enabled in your account/region
- Optional: `.env` — copy from `.env.example`

## Install

```bash
# Python 3.10+ required (see pyproject.toml). If `python3 --version` is older, use e.g. python3.11 -m venv .venv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip   # editable install needs a recent pip (PEP 660)
pip install -e ".[dev]"
```

## Run

```bash
# From repo root with venv activated
uvicorn ai_ops_chat.main:app --host 0.0.0.0 --port 8000
```

- Drop **any non-hidden file** (any extension or none — treated as UTF-8 text) under `LOG_WATCH_DIR` (default `./logs_inbox`), including nested subfolders when `LOG_WATCH_RECURSIVE` is `true` (the default). Hidden names (leading `.`) and `Thumbs.db` / `desktop.ini` are skipped. Files are scanned using the `:::LF:::` record delimiter (not line breaks alone). Incremental state lives in `CHROMA_PERSIST_DIR/ingest_state.json` (if you previously indexed with an older newline-based scheme, delete that file and re-ingest).
- `GET /health` — Chroma document count and Azure OpenAI reachability.
- `POST /ai/chat` — JSON body `{"query": "...", "user_id": "..."}`; response includes `analysis_html` (agent summary in HTML).

### Web UI (Vite)

A split-view browser UI lives under `frontend/`: text query on the left, rendered HTML report on the right (via `POST /ai/chat`). With the API on port **8000**, run:

```bash
cd frontend && npm install && npm run dev
```

Open the URL Vite prints (default `http://127.0.0.1:5173`). The dev server proxies `/ai` and `/health` to `http://127.0.0.1:8000`. For a fixed API URL instead (no proxy), create `frontend/.env.local` with `VITE_API_BASE_URL=http://127.0.0.1:8000` and set backend `CORS_ORIGINS` to include your UI origin (see `.env.example`).

### Log line format

Logical **records** end at the delimiter `:::LF:::`. The first line of a record is pipe-delimited (10 fields):

`field0|field1|...|field8|message`

Examples use `field0` like `[ERROR]` and a structured `message` segment with bracketed metadata and plain English text before `:::LF:::`. Optional text after `:::LF:::` (e.g. `:::LF::::` plus a colon and JVM stack frames) is kept in the **same record** until the next line that begins a new log with `[LEVEL]|`. Chroma documents embed the extracted English line plus any stack trace for search.

## Configuration

Environment variables (see [`.env.example`](.env.example)) include `LOG_WATCH_DIR`, `LOG_WATCH_RECURSIVE`, `CHROMA_PERSIST_DIR`, `AZURE_API_URL`, `AZURE_API_BASE`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`, `AWS_REGION`, `BEDROCK_MODEL_ID`, and `AGENT_NAME`.

**Important:** Use the same `AZURE_OPENAI_EMBED_DEPLOYMENT` for all ingestion after you start indexing; changing the deployment/model without re-ingesting will hurt retrieval quality.

If startup logs show `httpx.ReadTimeout` against Azure OpenAI, verify that `AZURE_API_URL` and `AZURE_OPENAI_API_KEY` are correct and that the embedding deployment exists. Raise `AZURE_OPENAI_EMBED_TIMEOUT_SECONDS` (default 600) if large batches time out, or lower **`AZURE_OPENAI_EMBED_BATCH_SIZE`** (default 16) so each request stays smaller. Hidden files like `.DS_Store` are not ingested. A failed initial scan is logged but the API still starts so you can fix connectivity and add or touch log files afterward.

## Tests

```bash
pytest
```

## Swagger endpoints

With the server running, interactive docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI) and [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) (ReDoc).

![OpenAPI / Swagger endpoints](swagger-endpoints.png)

![OpenAPI / Swagger Health check](health-check.png)

![OpenAPI / Swagger Query](query.png)

## Root Cause Analysis (HMTL)

Query outpurt report,

![RCA Output html](rca_output.html)
