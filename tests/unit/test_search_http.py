import httpx
import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import BudgetExceeded
from slrresearch.search.http import HttpClient, RateLimiter, SearchError


def make_http(handler, trace=None, max_calls=20, **kw):
    sleeps = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    http = HttpClient(client, Budget(max_calls=max_calls), trace, limiter=RateLimiter({}, sleep=sleeps.append),
                      sleep=sleeps.append, **kw)
    return http, sleeps


def test_success_first_try():
    http, sleeps = make_http(lambda r: httpx.Response(200, json={"ok": 1}))
    assert http.get("crossref", "https://x.test/a").json() == {"ok": 1} and sleeps == []


def test_retry_429_then_success_with_backoff():
    answers = [httpx.Response(429), httpx.Response(429), httpx.Response(200, json={})]
    http, sleeps = make_http(lambda r: answers.pop(0))
    assert http.get("semanticscholar", "https://x.test").status_code == 200
    assert sleeps == [2.0, 4.0]  # backoff exponentiel 2 s puis 4 s


def test_retry_after_header_is_honored_and_capped():
    answers = [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(503, headers={"retry-after": "999"}), httpx.Response(200)]
    http, sleeps = make_http(lambda r: answers.pop(0), max_wait=60.0)
    http.get("openalex", "https://x.test")
    assert sleeps == [7.0, 60.0]


def test_gives_up_after_retries_and_reports_429():
    http, sleeps = make_http(lambda r: httpx.Response(429), max_retries=2)
    with pytest.raises(SearchError) as err:
        http.get("semanticscholar", "https://x.test")
    assert err.value.status == 429 and "3 tentatives" in str(err.value) and len(sleeps) == 2


def test_client_error_is_not_retried():
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(404)
    http, sleeps = make_http(handler)
    with pytest.raises(SearchError) as err:
        http.get("crossref", "https://x.test")
    assert err.value.status == 404 and len(calls) == 1 and sleeps == []


def test_timeout_is_retried_then_succeeds():
    state = {"n": 0}
    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectTimeout("boom https://x.test?api_key=SECRET")
        return httpx.Response(200)
    http, sleeps = make_http(handler)
    assert http.get("arxiv", "https://x.test").status_code == 200 and sleeps == [2.0]


def test_every_attempt_consumes_budget_and_budget_error_propagates():
    http, _ = make_http(lambda r: httpx.Response(429), max_calls=2, max_retries=5)
    with pytest.raises(BudgetExceeded):
        http.get("crossref", "https://x.test")
    assert http.budget.calls == 2


def test_secret_never_reaches_trace_or_error(trace):
    def handler(request):
        raise httpx.ConnectTimeout("failed https://x.test?api_key=SECRET")
    http, _ = make_http(handler, trace=trace, max_retries=1)
    with pytest.raises(SearchError) as err:
        http.get("openalex", "https://x.test", params={"api_key": "SECRET"})
    assert "SECRET" not in str(err.value)
    assert "SECRET" not in str(trace.read())


def test_rate_limiter_spaces_calls_per_source():
    now = {"t": 100.0}
    sleeps = []
    def sleep(s):
        sleeps.append(s)
        now["t"] += s
    limiter = RateLimiter({"arxiv": 3.0}, clock=lambda: now["t"], sleep=sleep)
    limiter.wait("arxiv")          # premier appel : pas d'attente
    now["t"] += 1.0
    limiter.wait("arxiv")          # 1 s écoulée -> attend 2 s
    limiter.wait("openalex")       # autre source : aucune attente
    assert sleeps == [2.0]