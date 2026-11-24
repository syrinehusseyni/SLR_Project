"""Test avec le VRAI LLM du .env (Ollama ou Groq). Lancer :
python -m pytest -m live tests/live/test_protocol_live.py -v -s
"""
import pytest

from slrresearch.core.config import load_settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.protocol.questions_stage import QuestionsStage
from slrresearch.protocol.stage import ProtocolStage

TOPIC = ("Utilisation des grands modèles de langage (LLM) pour automatiser le screening titre/résumé "
         "dans les revues systématiques de la littérature")


@pytest.mark.live
def test_protocol_with_real_llm(store):
    ctx = build_context(store, load_settings())
    assert QuestionsStage(TOPIC).run(ctx).status == "done"
    result = ProtocolStage(year_from=2018).run(ctx)
    print("\nstatut :", result.status, result.message)
    assert result.status == "waiting_human"
    protocol = store.read_artifact("protocol", Protocol)
    print("génération :", protocol.generation)
    for concept in protocol.concepts:
        print(concept.id, concept.label, concept.terms)
    for query in protocol.search_plan.queries:
        print(query.source, "|", query.query_param, "=", query.query, query.params)