from ai_ops_chat.parser import extract_english_message, parse_pipe_log_record, parse_pipe_log_line


def test_parse_full_line_with_lf_suffix():
    line = (
        "E|svc|pid1|t1|handle|host1|req-1|u1|12ms|"
        "[2025-01-01T00:00:00Z] [MyClass] [API] [Orders] [lbl] [org] [it] [usr] [rw] [k] [] "
        "Payment gateway timeout when calling vendor:::LF:::"
    )
    p = parse_pipe_log_line(line)
    assert p is not None
    assert p.code == "E"
    assert p.request_id == "req-1"
    assert p.english_message == "Payment gateway timeout when calling vendor"
    assert p.stack_trace == ""


def test_extract_english_without_lf_marker():
    msg = "[t] [c] [x] [y] Simple failure message"
    assert extract_english_message(msg) == "Simple failure message"


def test_invalid_line_returns_none():
    assert parse_pipe_log_line("a|b|c") is None
    assert parse_pipe_log_line("") is None


def test_esri_bracket_severity_and_stack_in_embedding_text():
    record = (
        "[ERROR]|2026-03-30T15:34:30,888|0|Realtime Analytic Manager|ERROR|RESTWriter-Worker||HOST||"
        "system||[2026-03-30T22:34:30.888Z] [com.esri.Foo] [] [] [] [] [] [] [] [] [KEY] [url] "
        "Unable to post to URL.:::LF:::: java.net.NoRouteToHostException: No route to host\n"
        "\tat java.base/java.lang.Thread.run(Unknown Source)"
    )
    p = parse_pipe_log_record(record)
    print("\n\n PARSED RECORD: \n\n")
    print(p.english_message)
    print("\n\n EMBEDDING TEXT: \n\n")
    print(p.text_for_embedding)
    assert p is not None
    assert p.code == "[ERROR]"
    embed = p.text_for_embedding
    assert "Unable to post to URL" in embed
    assert "NoRouteToHostException" in embed
    assert "\tat java.base" in embed
