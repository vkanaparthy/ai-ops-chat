from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb

from ai_ops_chat.embeddings import embed_texts
from ai_ops_chat.parser import ParsedLogLine, parse_pipe_log_record, parsed_to_chroma_metadata

logger = logging.getLogger(__name__)


class ChromaManager:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        ollama_base_url: str,
        ollama_embed_model: str,
        ollama_embed_timeout_seconds: float,
        list_logs_max_ids: int,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ollama_base = ollama_base_url
        self._ollama_model = ollama_embed_model
        self._ollama_timeout = ollama_embed_timeout_seconds
        self._list_max = list_logs_max_ids
        self._lock = threading.Lock()

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        print("Embedding documents:" + str(len(texts)))
        return embed_texts(
            texts,
            base_url=self._ollama_base,
            model=self._ollama_model,
            timeout=self._ollama_timeout,
        )

    def add_parsed_lines(
        self,
        source_path: str,
        lines: list[tuple[int, str]],
    ) -> int:
        """Ingest list of (record_index, text) pairs; returns count added."""
        parsed_rows: list[tuple[int, ParsedLogLine]] = []
        for record_index, text in lines:
            p = parse_pipe_log_record(text)
            print("\n\n PARSED RECORD: \n\n")
            print(p.english_message)
            print("\n\n EMBEDDING TEXT: \n\n")
            print(p.text_for_embedding)
            if p is None:
                logger.debug("skip bad record %s:%s", source_path, record_index)
                continue
            parsed_rows.append((record_index, p))
        if not parsed_rows:
            return 0

        docs = [p.text_for_embedding for _, p in parsed_rows]
        embeddings = self.embed(docs)
        ids = [f"{source_path}:{idx}" for idx, _ in parsed_rows]
        metadatas = [
            parsed_to_chroma_metadata(p, source_path=source_path, record_index=idx)
            for idx, p in parsed_rows
        ]
        with self._lock:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metadatas,
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
