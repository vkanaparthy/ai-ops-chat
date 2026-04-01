from __future__ import annotations

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ai_ops_chat.chroma_manager import ChromaManager
from ai_ops_chat.ingest_state import IngestState
from ai_ops_chat.parser import extract_complete_log_records

logger = logging.getLogger(__name__)

# When ``LogFolderWatcher(..., recursive=True)``, the filesystem observer and
# startup scan walk all subdirectories under ``watch_dir``.


_LOG_SUFFIXES = {".log", ".txt"}
_IGNORE_BASENAMES = {"thumbs.db", "desktop.ini"}


def _should_ingest_path(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name.startswith("."):
        return False
    if name.lower() in _IGNORE_BASENAMES:
        return False
    suf = path.suffix.lower()
    return suf in _LOG_SUFFIXES or suf == ""


def ingest_file(
    path: Path,
    *,
    watch_root: Path,
    chroma: ChromaManager,
    state: IngestState,
) -> int:
    """Read new complete log *records* (``:::LF:::``-delimited); returns rows ingested."""
    if not _should_ingest_path(path):
        return 0
    try:
        raw = path.read_bytes()
    except OSError as e:
        logger.warning("read failed %s: %s", path, e)
        return 0
    try:
        rel = str(path.resolve().relative_to(watch_root.resolve()))
    except ValueError:
        rel = str(path.resolve())

    offset = state.get_offset(rel)
    remainder = state.get_remainder(rel)
    record_count = state.get_record_count(rel)

    if len(raw) < offset:
        state.reset_file(rel)
        offset = 0
        remainder = ""
        record_count = 0

    chunk = raw[offset:].decode("utf-8", errors="replace")
    buf = remainder + chunk
    records, rem_new = extract_complete_log_records(buf)

    consumed_chars = len(buf) - len(rem_new)
    if consumed_chars > 0:
        consumed_from_chunk = max(0, consumed_chars - len(remainder))
        chunk_prefix = chunk[:consumed_from_chunk]
        new_offset = offset + len(chunk_prefix.encode("utf-8"))
    else:
        # No complete record yet; buffer entire new file bytes into remainder.
        new_offset = offset + len(chunk.encode("utf-8"))

    if not records:
        state.set_file_progress(
            rel,
            offset=new_offset,
            remainder=rem_new,
            record_count=record_count,
        )
        return 0

    numbered = [(record_count + i, rec) for i, rec in enumerate(records)]
    n = chroma.add_parsed_lines(rel, numbered)
    new_record_count = record_count + len(records)

    state.set_file_progress(
        rel,
        offset=new_offset,
        remainder=rem_new,
        record_count=new_record_count,
    )
    logger.info("ingested %s rows from %s", n, rel)
    return n


class _DebouncedLogHandler(FileSystemEventHandler):
    def __init__(
        self,
        *,
        watch_root: Path,
        chroma: ChromaManager,
        state: IngestState,
        debounce_s: float,
    ) -> None:
        self._watch_root = watch_root.resolve()
        self._chroma = chroma
        self._state = state
        self._debounce_s = debounce_s
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, path: Path) -> None:
        key = str(path.resolve())
        with self._lock:
            if key in self._timers:
                self._timers[key].cancel()

            def run() -> None:
                with self._lock:
                    self._timers.pop(key, None)
                ingest_file(path, watch_root=self._watch_root, chroma=self._chroma, state=self._state)

            t = threading.Timer(self._debounce_s, run)
            t.daemon = True
            self._timers[key] = t
            t.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        # Copies often use write-to-temp + rename into the inbox; only the move
        # into the watched tree is observed, not a create on the final name.
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", None)
        if dest:
            self._schedule(Path(dest))


class LogFolderWatcher:
    def __init__(
        self,
        *,
        watch_dir: Path,
        chroma: ChromaManager,
        ingest_state_path: Path,
        debounce_s: float,
        recursive: bool,
    ) -> None:
        self._watch_dir = watch_dir
        self._chroma = chroma
        self._state = IngestState(ingest_state_path)
        self._debounce_s = debounce_s
        self._recursive = recursive
        self._observer: Observer | None = None

    def start(self) -> None:
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        handler = _DebouncedLogHandler(
            watch_root=self._watch_dir,
            chroma=self._chroma,
            state=self._state,
            debounce_s=self._debounce_s,
        )
        self._observer = Observer()
        self._observer.schedule(handler, str(self._watch_dir), recursive=self._recursive)
        self._observer.start()
        logger.info("watching %s recursive=%s", self._watch_dir, self._recursive)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def scan_existing(self) -> None:
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        root = self._watch_dir.resolve()
        if self._recursive:
            paths = sorted(p for p in root.rglob("*") if p.is_file())
        else:
            paths = sorted(p for p in root.iterdir() if p.is_file())
        for p in paths:
            ingest_file(p, watch_root=root, chroma=self._chroma, state=self._state)
