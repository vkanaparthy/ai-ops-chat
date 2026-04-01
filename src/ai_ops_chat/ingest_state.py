from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IngestState:
    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._data: dict[str, Any] = {"files": {}}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            if "files" not in self._data:
                self._data["files"] = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("reset ingest state: %s", e)
            self._data = {"files": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get_offset(self, file_key: str) -> int:
        entry = self._data["files"].get(file_key) or {}
        return int(entry.get("offset", 0))

    def get_remainder(self, file_key: str) -> str:
        entry = self._data["files"].get(file_key) or {}
        return str(entry.get("remainder", ""))

    def get_record_count(self, file_key: str) -> int:
        entry = self._data["files"].get(file_key) or {}
        return int(entry.get("record_count", 0))

    def set_file_progress(
        self,
        file_key: str,
        *,
        offset: int,
        remainder: str,
        record_count: int,
    ) -> None:
        self._data["files"].setdefault(file_key, {})
        self._data["files"][file_key]["offset"] = offset
        self._data["files"][file_key]["remainder"] = remainder
        self._data["files"][file_key]["record_count"] = record_count
        self._save()

    def reset_file(self, file_key: str) -> None:
        self._data["files"][file_key] = {
            "offset": 0,
            "remainder": "",
            "record_count": 0,
        }
        self._save()
