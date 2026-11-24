import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import BudgetExceeded


def test_budget_calls_exhausted_raises():
    b = Budget(max_calls=2)
    b.consume()
    b.consume()
    with pytest.raises(BudgetExceeded, match="appels"):
        b.consume()


def test_budget_remaining_calls():
    b = Budget(max_calls=3)
    b.consume()
    assert b.remaining_calls() == 2


def test_budget_time_exhausted_uses_fake_clock():
    now = [0.0]
    b = Budget(max_calls=100, max_seconds=10, clock=lambda: now[0])
    b.consume()
    now[0] = 11.0  # on "avance" le temps sans attendre
    with pytest.raises(BudgetExceeded, match="temps"):
        b.consume()


def test_budget_tokens_exhausted():
    b = Budget(max_calls=100, max_tokens=50)
    b.consume(tokens=60)
    with pytest.raises(BudgetExceeded, match="tokens"):
        b.consume()


def test_budget_does_not_count_a_refused_call():
    b = Budget(max_calls=1)
    b.consume()
    with pytest.raises(BudgetExceeded):
        b.consume()
    assert b.calls == 1