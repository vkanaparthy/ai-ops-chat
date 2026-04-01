from __future__ import annotations

import re
from dataclasses import dataclass

LOG_RECORD_DELIMITER = ":::LF:::"

# Next log record starts with [LEVEL]| e.g. [ERROR]|
NEW_LOG_LINE_START = re.compile(r"^\[[^\]]+\]\|")

MESSAGE_STRUCTURE = re.compile(
    r"^(?:(?:\[[^\]]*\]\s+)+)(?P<english>.+?)(?:\s*:::LF:::)?\s*$",
    re.DOTALL,
)

NEXT_LOG_AFTER_NEWLINE = re.compile(r"\n(\[[^\]]+\]\|)")


@dataclass(frozen=True)
class ParsedLogLine:
    code: str
    source: str
    process: str
    thread: str
    method_name: str
    machine: str
    request_id: str
    user: str
    elapsed: str
    message_raw: str
    english_message: str
    stack_trace: str

    @property
    def text_for_embedding(self) -> str:
        """English summary plus stack trace (if any) for vector search."""
        t = self.english_message.strip()
        st = self.stack_trace.strip()
        if st:
            return f"{t}\n{st}"
        return t


def _skip_after_delimiter(s: str, j: int) -> int:
    while j < len(s) and s[j] in " \t\r\n:":
        j += 1
    return j


def _is_new_log_at(s: str, j: int) -> bool:
    if j >= len(s) or s[j] != "[":
        return False
    return bool(NEW_LOG_LINE_START.match(s[j:]))


def _find_next_log_record_start(s: str, j: int) -> int | None:
    m = NEXT_LOG_AFTER_NEWLINE.search(s, j)
    if not m:
        return None
    pos = m.start(1)
    if not NEW_LOG_LINE_START.match(s[pos:]):
        return None
    return pos


def extract_complete_log_records(buffer: str) -> tuple[list[str], str]:
    """Split *buffer* into complete logical log records delimited by ``:::LF:::``.

    Continuations (e.g. Java stack traces) after ``:::LF:::`` belong to the same
    record until the next line that starts a new ``[LEVEL]|`` log.

    Returns:
        (complete_records, remainder) — remainder is an incomplete prefix for a
        future chunk (no trailing partial record is emitted).
    """
    records: list[str] = []
    i = 0
    delim = LOG_RECORD_DELIMITER
    ld = len(delim)
    buf = buffer

    while i < len(buf):
        pos = buf.find(delim, i)
        if pos == -1:
            return (records, buf[i:])
        end_delim = pos + ld
        j = _skip_after_delimiter(buf, end_delim)

        if j >= len(buf):
            records.append(buf[i:end_delim].strip())
            return (records, "")

        if _is_new_log_at(buf, j):
            records.append(buf[i:end_delim].strip())
            i = j
            continue

        next_log = _find_next_log_record_start(buf, j)
        if next_log is None:
            return (records, buf[i:])
        rec = buf[i:next_log].strip()
        records.append(rec)
        i = next_log

    return (records, "")


def split_log_records_complete_file(text: str) -> list[str]:
    """Parse a full file into records (no remainder)."""
    recs, rem = extract_complete_log_records(text)
    if rem.strip():
        recs.append(rem.strip())
    return recs


def extract_english_message(message_field: str) -> str:
    raw = message_field.strip()
    if LOG_RECORD_DELIMITER in raw:
        before_lf, _, _ = raw.partition(LOG_RECORD_DELIMITER)
        body = before_lf.strip()
    else:
        body = raw

    m = MESSAGE_STRUCTURE.match(body)
    if m:
        return m.group("english").strip()

    return body.strip()


def parse_pipe_log_record(record: str) -> ParsedLogLine | None:
    """Parse one logical record (may contain newlines / stack trace)."""
    s = record.strip()
    if not s:
        return None

    lines = s.splitlines()
    first = lines[0]
    rest_lines = lines[1:]

    stack_suffix_first = ""
    if LOG_RECORD_DELIMITER in first:
        main_first, _, after_delim = first.partition(LOG_RECORD_DELIMITER)
        stack_suffix_first = after_delim.lstrip()
        pipe_source = main_first
    else:
        pipe_source = first

    parts = pipe_source.split("|", 9)
    if len(parts) != 10:
        return None
    (
        code,
        source,
        process,
        thread,
        method_name,
        machine,
        request_id,
        user,
        elapsed,
        message_raw,
    ) = parts

    english = extract_english_message(message_raw)
    if not english:
        english = message_raw.strip()

    stack_parts: list[str] = []
    if stack_suffix_first:
        stack_parts.append(stack_suffix_first)
    if rest_lines:
        stack_parts.extend(rest_lines)
    stack_trace = "\n".join(stack_parts)

    return ParsedLogLine(
        code=code,
        source=source,
        process=process,
        thread=thread,
        method_name=method_name,
        machine=machine,
        request_id=request_id,
        user=user,
        elapsed=elapsed,
        message_raw=message_raw,
        english_message=english,
        stack_trace=stack_trace,
    )


def parse_pipe_log_line(line: str) -> ParsedLogLine | None:
    """Backward-compatible: one physical line equals one logical record."""
    return parse_pipe_log_record(line)


def parsed_to_chroma_metadata(
    parsed: ParsedLogLine,
    *,
    source_path: str,
    record_index: int,
) -> dict[str, str]:
    return {
        "code": parsed.code,
        "source": parsed.source,
        "process": parsed.process,
        "thread": parsed.thread,
        "method_name": parsed.method_name,
        "machine": parsed.machine,
        "request_id": parsed.request_id,
        "user_col": parsed.user,
        "elapsed": parsed.elapsed,
        "log_file": source_path,
        "record_index": str(record_index),
    }
