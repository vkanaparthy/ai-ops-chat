from __future__ import annotations

import json
import logging

import httpx

logger = logging.getLogger(__name__)


def embed_texts_ollama(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    timeout: float = 600.0,
    max_chars: int = 2000,
) -> list[list[float]]:
    """Call Ollama `/api/embed` for one or more strings.

    Args:
        max_chars: Truncate each text to this many characters before sending.
            nomic-embed-text has an 8192-token context. Stack traces tokenize
            densely (punctuation = 1 token each), so 2000 chars is a safe
            conservative ceiling while still capturing the full semantic signal.
    """
    url = base_url.rstrip("/") + "/api/embed"
    truncated = [t[:max_chars] for t in texts]
    if any(len(t) < len(o) for t, o in zip(truncated, texts)):
        logger.debug(
            "embed_texts_ollama: truncated %d text(s) to %d chars",
            sum(1 for t, o in zip(truncated, texts) if len(t) < len(o)),
            max_chars,
        )
    payload = {"model": model, "input": truncated}
    limits = httpx.Timeout(
        connect=30.0,
        read=timeout,
        write=timeout,
        pool=timeout,
    )
    with httpx.Client(timeout=limits) as client:
        r = client.post(url, json=payload)
        if not r.is_success:
            logger.error(
                "Ollama embed error %s — body: %s",
                r.status_code,
                r.text[:500],
            )
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


def embed_texts_bedrock(
    texts: list[str],
    *,
    region_name: str,
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True,
) -> list[list[float]]:
    """Call Amazon Bedrock Titan Embed Text v2 for one or more strings.

    Titan v2 accepts one text per call; this function loops internally.

    Args:
        texts: Input strings to embed.
        region_name: AWS region, e.g. "us-west-2".
        model_id: Bedrock embedding model ID.
        dimensions: Output vector size (256, 512, or 1024 for Titan v2).
        normalize: Whether to L2-normalise the output vectors.
    """
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region_name)
    embeddings: list[list[float]] = []
    for text in texts:
        body = json.dumps({"inputText": text, "dimensions": dimensions, "normalize": normalize})
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        embedding = result.get("embedding")
        if embedding is None:
            logger.error("Unexpected Bedrock embed response keys: %s", result.keys())
            raise ValueError("Bedrock embed response missing embedding")
        embeddings.append(embedding)
    return embeddings
