from ai_ops_chat.chroma_manager import ChromaManager
from ai_ops_chat.ingest_state import IngestState
from ai_ops_chat.parser import extract_complete_log_records
from ai_ops_chat.watcher import _should_ingest_path, ingest_file


def test_extract_complete_single_esri_style_record():
    line = (
        "[ERROR]|2026-03-30T15:34:30,889|0|Realtime Analytic Manager|ERROR|thr||HOST||system||"
        "[2026-03-30T22:34:30.889Z] [com.esri.realtime.io.output.arcgis.SaveToRESTWorker] [unknown] [] [] "
        "[system] [] [] [user] [AGOL__QUERY_FEATURES_ERROR] [] Unable to query for features.:::LF::: "
    )
    recs, rem = extract_complete_log_records(line)
    assert len(recs) == 1
    assert "Unable to query for features" in recs[0]
    assert rem == ""


def test_extract_complete_record_with_stack_trace_before_next_log():
    main = (
        "[ERROR]|2026-03-30T15:34:30,888|0|Realtime Analytic Manager|ERROR|RESTWriter||HOST||system||"
        "[2026-03-30T22:34:30.888Z] [com.esri.Class] [output] [feat] [lbl] [org] [item] [user] [portaladmin] "
        "[admin] [AGOL__WRITER_UNABLE_TO_POST_TO_URL] [https://example.com/query] Unable to post to URL "
        "https://example.com/query.:::LF:::: java.net.NoRouteToHostException: No route to host\n"
        "\tat java.base/sun.nio.ch.Net.pollConnect(Native Method)\n"
        "\tat java.base/java.lang.Thread.run(Unknown Source)\n"
    )
    nxt = (
        "[ERROR]|2026-03-30T15:34:30,889|0|Realtime Analytic Manager|ERROR|other||HOST||system||"
        "[t] [c] [] [] [] [] [] [] [] [] [] Next log message.:::LF::: "
    )
    buf = main + nxt
    recs, rem = extract_complete_log_records(buf)
    assert len(recs) == 2
    assert "NoRouteToHostException" in recs[0]
    assert "\tat java.base" in recs[0]
    assert "Next log message" in recs[1]
    assert rem == ""


def test_extract_incomplete_buffer_returns_remainder():
    buf = "[ERROR]|...|...|msg.:::LF:::: java excep"
    recs, rem = extract_complete_log_records(buf)
    assert recs == []
    assert rem == buf


def test_ingest_file_accepts_delimited_record_without_trailing_newline(tmp_path):
    log_body = (
        "E|svc|pid|th|meth|m|req1|u|1ms|"
        "[t] [c] [x] [y] [l] [o] [i] [u] [a] [k] [] hello:::LF:::"
    )
    f = tmp_path / "test.log"
    f.write_text(log_body, encoding="utf-8")

    chroma = ChromaManager(
        persist_dir=tmp_path / "chroma",
        collection_name="testcol",
        ollama_base_url="http://127.0.0.1:99999",
        ollama_embed_model="dummy",
        ollama_embed_timeout_seconds=30.0,
        list_logs_max_ids=100,
    )
    state = IngestState(tmp_path / "state.json")
    chroma.embed = lambda texts: [[0.0, 0.0] for _ in texts]  # type: ignore[method-assign]

    n = ingest_file(f, watch_root=tmp_path, chroma=chroma, state=state)
    assert n == 1
    assert chroma.count_documents() == 1


def test_should_skip_dotfiles_and_ds_store(tmp_path):
    assert _should_ingest_path(tmp_path / ".DS_Store") is False
    f = tmp_path / ".DS_Store"
    f.write_bytes(b"x")
    assert _should_ingest_path(f) is False
    ok = tmp_path / "app.log"
    ok.write_text("x", encoding="utf-8")
    assert _should_ingest_path(ok) is True
