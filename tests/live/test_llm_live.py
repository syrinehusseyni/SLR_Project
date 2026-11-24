import pytest

from slrresearch.core.config import build_model, load_settings
from slrresearch.core.llm import LLMClient


@pytest.mark.live
def test_real_llm_answers():
    """Vrai appel au LLM du .env (Ollama ou Groq). Lance avec : pytest -m live tests/live -v -s"""
    settings = load_settings()
    model = build_model(settings, "screening")
    client = LLMClient(model, settings.model_for("screening"), "screening", max_retries=1)
    resp = client.call("Reponds uniquement par le mot : OK")
    print(f"\nprovider={settings.llm_provider} model={resp.model} "
          f"temps={resp.seconds:.1f}s tokens={resp.tokens} reponse={resp.text!r}")
    assert resp.text.strip()