"""Budget d'execution : appels, tokens, temps.

Idee du cours : "le LLM propose, le programme controle". Un agent ne decide pas
seul quand s'arreter : le programme coupe quand le budget est epuise.
"""
import time
from typing import Callable

from slrresearch.core.errors import BudgetExceeded


class Budget:
    def __init__(
        self,
        max_calls: int,
        max_seconds: float | None = None,
        max_tokens: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_calls < 0:
            raise ValueError("max_calls doit etre >= 0")
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.max_tokens = max_tokens
        self._clock = clock  # injectable : permet de tester le temps sans attendre
        self._start = clock()
        self.calls = 0
        self.tokens = 0

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def check(self) -> None:
        """Leve BudgetExceeded si une limite est deja depassee."""
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"budget d'appels epuise ({self.calls}/{self.max_calls})")
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            raise BudgetExceeded(f"budget de temps epuise ({self.elapsed:.1f}s/{self.max_seconds}s)")
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"budget de tokens epuise ({self.tokens}/{self.max_tokens})")

    def consume(self, calls: int = 1, tokens: int = 0) -> None:
        """Verifie le budget AVANT d'agir, puis enregistre la consommation."""
        self.check()
        self.calls += calls
        self.tokens += tokens

    def add_tokens(self, tokens: int) -> None:
        """Enregistre des tokens APRES un appel reussi, sans verifier le budget
        (sinon on perdrait une reponse deja payee)."""
        self.tokens += tokens

        
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls)