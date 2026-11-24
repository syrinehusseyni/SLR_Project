import pytest

from slrresearch.core.config import build_model, load_settings
from slrresearch.core.llm import LLMClient
from slrresearch.protocol.questions import generate_research_questions

TOPIC = ("Utilisation des grands modèles de langage (LLM) pour automatiser le screening "
         "titre/résumé dans les revues systématiques de la littérature")


@pytest.mark.live
def test_real_llm_generates_valid_questions():
    """Vrai LLM du .env. Lance avec : python -m pytest -m live tests/live/test_questions_live.py -v -s"""
    settings = load_settings()
    llm = LLMClient(build_model(settings, "planner"), settings.model_for("planner"), "planner", max_retries=1)
    outcome = generate_research_questions(llm, TOPIC)
    print(f"\nprovider={settings.llm_provider} status={outcome.status} repairs={outcome.repair_attempts}")
    for q in outcome.questions:
        print(f"  {q.identifier}. {q.question}")
    if outcome.status == "human_required":
        print(f"  ERREUR : {outcome.error}\n  TEXTE : {outcome.text!r}")
    assert outcome.status != "human_required", outcome.error