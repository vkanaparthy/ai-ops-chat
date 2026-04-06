from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_ops_chat.agent_runner import run_rca_agent
from ai_ops_chat.chroma_manager import ChromaManager
from ai_ops_chat.config import Settings, get_settings
from ai_ops_chat.watcher import LogFolderWatcher

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    user_id: str
    query: str
    analysis_html: str
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    chroma_documents: int
    ollama_ok: bool
    detail: dict[str, Any] | None = None


_chroma: ChromaManager | None = None
_watcher: LogFolderWatcher | None = None
_settings: Settings | None = None


def get_chroma() -> ChromaManager:
    if _chroma is None:
        raise RuntimeError("Chroma not initialized")
    return _chroma


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chroma, _watcher, _settings
    _settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _chroma = ChromaManager(
        persist_dir=_settings.chroma_persist_dir,
        collection_name=_settings.chroma_collection,
        ollama_base_url=_settings.ollama_base_url,
        ollama_embed_model=_settings.ollama_embed_model,
        ollama_embed_timeout_seconds=_settings.ollama_embed_timeout_seconds,
        ollama_embed_batch_size=_settings.ollama_embed_batch_size,
        list_logs_max_ids=_settings.list_logs_max_ids,
    )
    ingest_state = _settings.chroma_persist_dir / "ingest_state.json"
    _watcher = LogFolderWatcher(
        watch_dir=_settings.log_watch_dir,
        chroma=_chroma,
        ingest_state_path=ingest_state,
        debounce_s=_settings.watch_debounce_seconds,
        recursive=_settings.log_watch_recursive,
    )
    try:
        _watcher.scan_existing()
    except Exception:
        logger.exception(
            "Initial log scan failed (Ollama slow/unreachable, timeout, or bad files?). "
            "Ensure `ollama serve` is running and `%s` is pulled. The server will still run; "
            "drop or touch logs after Ollama is ready to ingest.",
            _settings.ollama_embed_model,
        )
    _watcher.start()
    yield
    if _watcher:
        _watcher.stop()
    _chroma = None
    _watcher = None
    _settings = None


app = FastAPI(title="AI Ops Chat — RCA API", lifespan=lifespan)


@app.post("/ai/chat", response_model=ChatResponse)
def ai_chat(body: ChatRequest) -> ChatResponse:
    settings = _settings or get_settings()
    chroma = get_chroma()
    logger.info("chat user_id=%s query_chars=%s", body.user_id, len(body.query))
    try:
        html = run_rca_agent(body.query, chroma, settings)
    except Exception as e:
        logger.exception("agent failed")
        raise HTTPException(status_code=502, detail={"error": str(e)}) from e
    return ChatResponse(user_id=body.user_id, query=body.query, analysis_html=html, status="ok")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = _settings or get_settings()
    chroma = get_chroma()
    n = chroma.count_documents()
    ollama_ok = False
    detail: dict[str, Any] = {}
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(settings.ollama_base_url.rstrip("/") + "/api/tags")
            ollama_ok = r.is_success
            detail["ollama_status"] = r.status_code
    except Exception as e:
        detail["ollama_error"] = str(e)
    overall = "ok" if ollama_ok else "degraded"
    return HealthResponse(
        status=overall,
        chroma_documents=n,
        ollama_ok=ollama_ok,
        detail=detail or None,
    )
