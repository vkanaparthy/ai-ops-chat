from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def embed_texts(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    timeout: float = 600.0,
) -> list[list[float]]:
    """Call Ollama `/api/embed` for one or more strings."""
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": texts}
    limits = httpx.Timeout(
        connect=30.0,
        read=timeout,
        write=timeout,
        pool=timeout,
    )
    with httpx.Client(timeout=limits) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    embeddings = data.get("embeddings")
    if embeddings is None:
        # single-input shape
        one = data.get("embedding")
        if one is not None:
            return [one]
        logger.error("Unexpected Ollama embed response keys: %s", data.keys())
        raise ValueError("Ollama embed response missing embeddings")
    if len(embeddings) != len(texts):
        raise ValueError("Ollama returned unexpected embedding count")
    return embeddings
