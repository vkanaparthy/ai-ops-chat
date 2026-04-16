from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb

from ai_ops_chat.embeddings import embed_texts_bedrock, embed_texts_ollama
from ai_ops_chat.parser import ParsedLogLine, parse_pipe_log_record, parsed_to_chroma_metadata

logger = logging.getLogger(__name__)


class ChromaManager:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        embed_provider: str = "ollama",
        # Ollama settings
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_embed_model: str = "nomic-embed-text",
        ollama_embed_timeout_seconds: float = 600.0,
        ollama_embed_batch_size: int = 8,
        # Bedrock settings
        aws_region: str = "us-west-2",
        bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0",
        bedrock_embed_dimensions: int = 1024,
        bedrock_embed_batch_size: int = 16,
        list_logs_max_ids: int = 10_000,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._provider = embed_provider.lower()
        # Ollama
        self._ollama_base = ollama_base_url
        self._ollama_model = ollama_embed_model
        self._ollama_timeout = ollama_embed_timeout_seconds
        self._ollama_batch_size = max(1, ollama_embed_batch_size)
        # Bedrock
        self._aws_region = aws_region
        self._bedrock_embed_model_id = bedrock_embed_model_id
        self._bedrock_embed_dimensions = bedrock_embed_dimensions
        self._bedrock_batch_size = max(1, bedrock_embed_batch_size)

        self._list_max = list_logs_max_ids
        self._lock = threading.Lock()

    @property
    def embed_provider(self) -> str:
        return self._provider

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._provider == "bedrock":
            return self._embed_bedrock(texts)
        return self._embed_ollama(texts)

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        bs = self._ollama_batch_size
        kwargs = dict(
            base_url=self._ollama_base,
            model=self._ollama_model,
            timeout=self._ollama_timeout,
        )
        if len(texts) <= bs:
            return embed_texts_ollama(texts, **kwargs)
        num_req = (len(texts) + bs - 1) // bs
        logger.info(
            "Ollama embed: %d text(s) in %d batch(es) of up to %d (model=%s)",
            len(texts),
            num_req,
            bs,
            self._ollama_model,
        )
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            logger.debug("Ollama embed batch %d-%d (%d texts)", i, i + len(batch) - 1, len(batch))
            out.extend(embed_texts_ollama(batch, **kwargs))
        return out

    def _embed_bedrock(self, texts: list[str]) -> list[list[float]]:
        bs = self._bedrock_batch_size
        kwargs = dict(
            region_name=self._aws_region,
            model_id=self._bedrock_embed_model_id,
            dimensions=self._bedrock_embed_dimensions,
        )
        num_req = (len(texts) + bs - 1) // bs
        if num_req > 1:
            logger.info(
                "Bedrock embed: %d text(s) in %d batch(es) of up to %d (model=%s)",
                len(texts),
                num_req,
                bs,
                self._bedrock_embed_model_id,
            )
        out: list[list[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            logger.debug("Bedrock embed batch %d-%d (%d texts)", i, i + len(batch) - 1, len(batch))
            out.extend(embed_texts_bedrock(batch, **kwargs))
        return out

    def add_parsed_lines(
        self,
        source_path: str,
        lines: list[tuple[int, str]],
    ) -> int:
        """Ingest list of (record_index, text) pairs; returns count added."""
        n_in = len(lines)
        parsed_rows: list[tuple[int, ParsedLogLine]] = []
        for record_index, text in lines:
            p = parse_pipe_log_record(text)
            if p is None:
                logger.debug("skip bad record %s:%s", source_path, record_index)
                continue
            parsed_rows.append((record_index, p))
        n_skip = n_in - len(parsed_rows)
        if n_skip:
            logger.info(
                "ingest Chroma parse %s: kept=%d skipped_unparseable=%d",
                source_path,
                len(parsed_rows),
                n_skip,
            )
        elif parsed_rows:
            logger.debug("ingest Chroma parse %s: kept=%d", source_path, len(parsed_rows))

        if not parsed_rows:
            return 0

        docs = [p.text_for_embedding for _, p in parsed_rows]
        idx_min = min(idx for idx, _ in parsed_rows)
        idx_max = max(idx for idx, _ in parsed_rows)
        batch_size = self._bedrock_batch_size if self._provider == "bedrock" else self._ollama_batch_size
        logger.info(
            "ingest Chroma %s: embed %d document(s) id_range=[%s:%d..%d] batch_size=%d",
            source_path,
            len(docs),
            source_path,
            idx_min,
            idx_max,
            batch_size,
        )
        embeddings = self.embed(docs)
        ids = [f"{source_path}:{idx}" for idx, _ in parsed_rows]
        metadatas = [
            parsed_to_chroma_metadata(p, source_path=source_path, record_index=idx)
            for idx, p in parsed_rows
        ]
        with self._lock:
            total_before = self._collection.count()
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas,
            )
            total_after = self._collection.count()
        logger.info(
            "ingest Chroma %s: added=%d collection_count %d -> %d",
            source_path,
            len(ids),
            total_before,
            total_after,
        )
        return len(ids)

    def count_documents(self) -> int:
        with self._lock:
            return self._collection.count()

    def list_log_ids(self) -> list[str]:
        with self._lock:
            res = self._collection.get(include=[])
        ids = list(res.get("ids") or [])
        ids_sorted = sorted(set(ids))[: self._list_max]
        return ids_sorted

    def search_logs(self, query: str, top_k: int) -> list[dict]:
        q = query.strip()
        if not q:
            return []
        q_emb = self.embed([q])
        n = max(1, min(top_k, 100))
        with self._lock:
            result = self._collection.query(
                query_embeddings=q_emb,
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        out: list[dict] = []
        doc_lists = result.get("documents") or [[]]
        meta_lists = result.get("metadatas") or [[]]
        dist_lists = result.get("distances") or [[]]
        for i in range(len(doc_lists[0])):
            out.append(
                {
                    "document": doc_lists[0][i],
                    "metadata": meta_lists[0][i],
                    "distance": dist_lists[0][i] if dist_lists[0] else None,
                }
            )
        return out
