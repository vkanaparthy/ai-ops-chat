from __future__ import annotations

import json
from typing import Any, Callable

from strands import tool

from ai_ops_chat.chroma_manager import ChromaManager


def build_tools(
    chroma: ChromaManager,
    *,
    default_top_k: int,
) -> list[Callable[..., Any]]:
    """Create Strands @tool callables bound to a ChromaManager instance."""

    @tool
    def list_logs() -> str:
        """List all unique log document IDs available in the ChromaDB collection.

        Call this tool only once when the user asks what log data exists or what is indexed.
        Do not call it for root-cause or error questions; use search_logs instead.

        Returns:
            Newline-separated log IDs (file path and line number), or a note if empty.
        """
        ids = chroma.list_log_ids()
        if not ids:
            return "No log documents are indexed yet."
        return "\n".join(ids)

    @tool
    def search_logs(query: str, top_k: int = default_top_k) -> str:
        """Search indexed logs using semantic similarity over embedded English log messages.

        Use this for specific questions about errors, failures, root causes, or symptoms.
        Return value includes full matching document text and metadata for analysis.

        Args:
            query: Natural language search query (symptoms, error text, component, etc.).
            top_k: Max results to return (default from server configuration).

        Returns:
            JSON list of matches with document text, metadata, and distance scores.
        """
        rows = chroma.search_logs(query, top_k)
        return json.dumps(rows, indent=2)

    return [list_logs, search_logs]
