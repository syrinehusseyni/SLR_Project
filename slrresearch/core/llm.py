"""Appel LLM unique : retry avec backoff, budget, journal (trace) et comptage des tokens.

Tous les agents passent par LLMClient.call(). Ainsi les echecs reseau, les limites de
debit et la consommation sont geres a UN seul endroit, pas recopies dans chaque agent.
"""
import time
from dataclasses import dataclass
from typing import Any, Callable

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import LLMError, SLRError
from slrresearch.core.trace import Trace


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens: int
    seconds: float
    attempts: int


def _extract_text(raw: Any) -> str:
    # Un vrai modele LangChain renvoie un message (.content) ; le FakeLLM renvoie un str.
    content = getattr(raw, "content", raw)
    return content if isinstance(content, str) else str(content)


def _extract_tokens(raw: Any) -> int:
    usage = getattr(raw, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or 0)
    return 0


class LLMClient:
    def __init__(
        self,
        model: Any,
        model_name: str,
        role: str,
        budget: Budget | None = None,
        trace: Trace | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.model_name = model_name
        self.role = role
        self.budget = budget
        self.trace = trace
        self.max_retries = max_retries  # essais SUPPLEMENTAIRES apres le premier
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep  # injectable : les tests n'attendent pas vraiment
        self._clock = clock
        self.calls = 0
        self.total_tokens = 0

    def _log(self, event: str, **data: Any) -> None:
        if self.trace:
            self.trace.log(self.role, event, model=self.model_name, **data)

    def call(self, prompt: str) -> LLMResponse:
        attempts_allowed = self.max_retries + 1
        for attempt in range(1, attempts_allowed + 1):
            if self.budget:
                self.budget.consume()  # BudgetExceeded n'est JAMAIS reessaye : on s'arrete
            start = self._clock()
            try:
                raw = self.model.invoke(prompt)
            except SLRError:
                raise
            except Exception as exc:  # erreur reseau, 429, timeout...
                self._log("llm_error", attempt=attempt, error=f"{type(exc).__name__}: {exc}")
                if attempt == attempts_allowed:
                    raise LLMError(
                        f"appel LLM ({self.model_name}) echoue apres {attempts_allowed} essai(s) : {exc}"
                    ) from exc
                self._sleep(self.backoff_seconds * 2 ** (attempt - 1))  # 1s, 2s, 4s...
                continue

            text = _extract_text(raw)
            tokens = _extract_tokens(raw)
            seconds = self._clock() - start
            self.calls += 1
            self.total_tokens += tokens
            if self.budget:
                self.budget.add_tokens(tokens)
            self._log(
                "llm_call",
                attempt=attempt,
                seconds=round(seconds, 3),
                tokens=tokens,
                prompt_chars=len(prompt),
                response_chars=len(text),
            )
            return LLMResponse(text=text, model=self.model_name, tokens=tokens, seconds=seconds, attempts=attempt)

        raise LLMError("etat impossible")  # pragma: no cover