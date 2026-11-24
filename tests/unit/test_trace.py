from slrresearch.core.trace import Trace
from tests.fakes import FakeLLM


def test_trace_appends_without_overwriting(trace):
    trace.log("search", "query_sent", source="openalex")
    trace.log("search", "got_results", n=5)
    events = trace.read()
    assert [e["event"] for e in events] == ["query_sent", "got_results"]
    assert events[1]["data"]["n"] == 5


def test_trace_survives_reopening(store):
    Trace(store.trace_path).log("a", "e1")
    Trace(store.trace_path).log("a", "e2")  # nouvelle instance = meme fichier
    assert len(Trace(store.trace_path).read()) == 2


def test_trace_does_not_crash_on_unserializable(trace):
    trace.log("a", "weird", obj=object())
    assert trace.read()[0]["event"] == "weird"


def test_trace_empty_when_no_file(tmp_path):
    assert Trace(tmp_path / "x" / "t.jsonl").read() == []


def test_fakellm_counts_calls_and_fails_when_exhausted():
    llm = FakeLLM(["a", "b"])
    assert llm.invoke("p1") == "a"
    assert llm.invoke("p2") == "b"
    assert llm.calls == 2
    try:
        llm.invoke("p3")
    except AssertionError:
        pass
    else:
        raise AssertionError("FakeLLM aurait du refuser un 3e appel")