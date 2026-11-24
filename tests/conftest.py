import pytest

from slrresearch.core.store import RunStore
from slrresearch.core.trace import Trace


@pytest.fixture
def store(tmp_path):
    """Un RunStore dans un dossier temporaire (supprime apres le test)."""
    return RunStore(tmp_path, "run_test")


@pytest.fixture
def trace(store):
    return Trace(store.trace_path)