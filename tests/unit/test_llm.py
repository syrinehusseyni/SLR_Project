from types import SimpleNamespace

import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import BudgetExceeded, LLMError
from slrresearch.core.llm import LLMClient
from tests.fakes import FakeLLM, FlakyLLM


def make_client(model, **kw):
    slept = []
    kw.setdefault("sleep", slept.append)  # on enregistre les attentes au lieu d'attendre
    client = LLMClient(model, model_name="fake-model", role="screening", **kw)
    return client, slept


def test_success_returns_text_and_counts_call():
    llm = FakeLLM(["bonjour"])
    client, _ = make_client(llm)
    resp = client.call("salut")
    assert resp.text == "bonjour"
    assert resp.attempts == 1
    assert llm.calls == 1 and client.calls == 1


def test_reads_content_and_tokens_from_langchain_style_message():
    msg = SimpleNamespace(content="reponse", usage_metadata={"total_tokens": 42})
    budget = Budget(max_calls=5)
    client, _ = make_client(FakeLLM([msg]), budget=budget)
    resp = client.call("p")
    assert resp.text == "reponse"
    assert resp.tokens == 42
    assert client.total_tokens == 42 and budget.tokens == 42


def test_retry_then_success_uses_exponential_backoff():
    flaky = FlakyLLM(failures=2, answer="enfin")
    client, slept = make_client(flaky, max_retries=2, backoff_seconds=1.0)
    resp = client.call("p")
    assert resp.text == "enfin"
    assert resp.attempts == 3
    assert slept == [1.0, 2.0]


def test_all_attempts_fail_raises_llmerror():
    flaky = FlakyLLM(failures=99)
    client, slept = make_client(flaky, max_retries=2)
    with pytest.raises(LLMError, match="3 essai"):
        client.call("p")
    assert flaky.calls == 3
    assert len(slept) == 2  # pas d'attente apres le dernier echec


def test_each_attempt_consumes_budget():
    budget = Budget(max_calls=10)
    client, _ = make_client(FlakyLLM(failures=2), budget=budget, max_retries=2)
    client.call("p")
    assert budget.calls == 3


def test_budget_exhausted_stops_without_calling_model():
    llm = FakeLLM(["a", "b"])
    client, _ = make_client(llm, budget=Budget(max_calls=1))
    client.call("p1")
    with pytest.raises(BudgetExceeded):
        client.call("p2")
    assert llm.calls == 1  # le 2e appel n'a jamais atteint le modele


def test_budget_error_is_not_retried():
    llm = FakeLLM(["a"])
    client, slept = make_client(llm, budget=Budget(max_calls=0), max_retries=5)
    with pytest.raises(BudgetExceeded):
        client.call("p")
    assert llm.calls == 0 and slept == []


def test_calls_and_errors_are_written_to_trace(trace):
    client, _ = make_client(FlakyLLM(failures=1), trace=trace, max_retries=1)
    client.call("un prompt")
    events = [e["event"] for e in trace.read()]
    assert events == ["llm_error", "llm_call"]
    last = trace.read()[-1]
    assert last["agent"] == "screening"
    assert last["data"]["model"] == "fake-model"
    assert last["data"]["prompt_chars"] == len("un prompt")