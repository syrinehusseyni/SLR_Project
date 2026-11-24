"""Couche HTTP commune aux 4 sources : budget d'appels, espacement, retry avec backoff sur 429/5xx.

Aucun secret (clé API) n'est écrit dans la trace ni dans les messages d'erreur.
"""
import time

import httpx

from slrresearch.core.errors import SLRError

# Espacement minimal entre deux appels à la même source (secondes), d'après leurs règles d'usage.
MIN_INTERVALS = {"openalex": 0.2, "crossref": 0.2, "semanticscholar": 1.1, "arxiv": 3.0}
RETRYABLE = {429, 500, 502, 503, 504}


class SearchError(SLRError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class MissingCredential(SearchError):
    """Une clé obligatoire manque : la source est ignorée, pas le run entier."""


class RateLimiter:
    def __init__(self, intervals=None, clock=time.monotonic, sleep=time.sleep):
        self.intervals = dict(MIN_INTERVALS if intervals is None else intervals)
        self.clock, self.sleep = clock, sleep
        self._last: dict[str, float] = {}

    def wait(self, source: str) -> None:
        interval = self.intervals.get(source, 0.0)
        last = self._last.get(source)
        if interval and last is not None:
            remaining = interval - (self.clock() - last)
            if remaining > 0:
                self.sleep(remaining)
        self._last[source] = self.clock()


class HttpClient:
    def __init__(self, client: httpx.Client, budget, trace=None, limiter=None, max_retries=3,
                 backoff_seconds=2.0, max_wait=60.0, sleep=time.sleep):
        self.client, self.budget, self.trace = client, budget, trace
        self.limiter = limiter or RateLimiter(sleep=sleep)
        self.max_retries, self.backoff, self.max_wait, self.sleep = max_retries, backoff_seconds, max_wait, sleep

    def get(self, source: str, url: str, params=None, headers=None) -> httpx.Response:
        last_status, last_reason, delay = None, "", 0.0
        for attempt in range(1, self.max_retries + 2):
            self.budget.consume()  # BudgetExceeded remonte : on ne la réessaie jamais
            self.limiter.wait(source)
            try:
                response = self.client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_status, last_reason = None, type(exc).__name__  # jamais str(exc) : l'URL peut contenir la clé
                delay = self._backoff(attempt)
                self._log(source, url, None, attempt, last_reason)
            else:
                status = response.status_code
                self._log(source, url, status, attempt)
                if status < 400:
                    return response
                if status not in RETRYABLE:
                    raise SearchError(f"{source} : HTTP {status}", status=status)
                last_status, last_reason = status, f"HTTP {status}"
                delay = self._retry_after(response) or self._backoff(attempt)
            if attempt <= self.max_retries:
                self.sleep(min(delay, self.max_wait))
        raise SearchError(f"{source} : échec après {self.max_retries + 1} tentatives ({last_reason})", status=last_status)

    def _backoff(self, attempt: int) -> float:
        return self.backoff * 2 ** (attempt - 1)

    @staticmethod
    def _retry_after(response) -> float:
        raw = response.headers.get("retry-after", "")
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0

    def _log(self, source, url, status, attempt, reason=""):
        if self.trace is not None:
            self.trace.log("search", "http_get", source=source, url=url, status=status, attempt=attempt, reason=reason)