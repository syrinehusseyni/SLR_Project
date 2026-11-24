import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import BudgetExceeded
from slrresearch.core.llm import LLMClient
from slrresearch.core.schemas.protocol import QuestionsOutcome
from slrresearch.protocol.questions import (
    build_prompt,
    build_repair_prompt,
    generate_research_questions,
    parse_questions,
)
from tests.fakes import FakeLLM

TOPIC = "LLM pour le screening des revues systématiques"
GOOD = (
    "RQ1. Quels LLM sont utilisés pour le screening titre/résumé ?\n"
    "RQ2. Quelles métriques évaluent la qualité du screening automatisé ?\n"
    "RQ3. Quelles limites sont rapportées pour ces approches ?"
)
INTRO = "Voici les trois questions :\n" + GOOD


def client_with(responses, **kw):
    fake = FakeLLM(responses)
    return LLMClient(fake, model_name="fake", role="planner", sleep=lambda s: None, **kw), fake


# -- parse_questions : controle deterministe --------------------------------------
def test_parse_valid_three_questions():
    qs = parse_questions(GOOD)
    assert [q.identifier for q in qs] == ["RQ1", "RQ2", "RQ3"]


def test_parse_accepts_other_count_when_asked():
    two = "\n".join(GOOD.splitlines()[:2])
    assert len(parse_questions(two, n=2)) == 2


@pytest.mark.parametrize(
    "bad, message",
    [
        ("", "vide"),
        ("   \n  ", "vide"),
        (INTRO, "non conforme"),  # phrase d'introduction du LLM
        ("RQ1. Trop court ?\nRQ2. x\nRQ3. y", "non conforme"),  # < 15 caracteres
        ("RQ1 Question sans point final de numero valide ?", "non conforme"),
    ],
)
def test_parse_rejects_malformed_text(bad, message):
    with pytest.raises(ValueError, match=message):
        parse_questions(bad)


def test_parse_rejects_wrong_order_and_missing_questions():
    lines = GOOD.splitlines()
    with pytest.raises(ValueError, match="Identifiants attendus"):
        parse_questions("\n".join([lines[0], lines[2], lines[1]]))
    with pytest.raises(ValueError, match="Identifiants attendus"):
        parse_questions("\n".join(lines[:2]))  # 2 questions au lieu de 3


def test_parse_rejects_duplicates_even_with_case_and_spaces_differences():
    dup = (
        "RQ1. Quels LLM sont utilisés pour le screening ?\n"
        "RQ2. quels  LLM sont utilisés pour le SCREENING ?\n"
        "RQ3. Quelles limites sont rapportées pour ces approches ?"
    )
    with pytest.raises(ValueError, match="differentes"):
        parse_questions(dup)


def test_invalid_question_count_refused():
    with pytest.raises(ValueError, match="entre 1 et 5"):
        parse_questions(GOOD, n=9)


# -- prompts ---------------------------------------------------------------------------
def test_prompt_contains_topic_rules_and_examples():
    p = build_prompt(TOPIC)
    assert TOPIC in p
    assert "exactement 3" in p
    assert "N'invente ni référence" in p
    assert "RQ3." in p and "RQ4." not in p  # exemples limites a n lignes


def test_repair_prompt_contains_error_and_bad_text():
    p = build_repair_prompt("texte casse", "Ligne non conforme : abc")
    assert "Ligne non conforme : abc" in p and "texte casse" in p


# -- generate_research_questions ------------------------------------------------------
def test_accepted_on_first_try_uses_one_call():
    llm, fake = client_with([GOOD])
    outcome = generate_research_questions(llm, TOPIC)
    assert outcome.status == "accepted"
    assert outcome.repair_attempts == 0
    assert len(outcome.questions) == 3
    assert fake.calls == 1


def test_repaired_after_one_bad_output():
    llm, fake = client_with([INTRO, GOOD])
    outcome = generate_research_questions(llm, TOPIC)
    assert outcome.status == "repaired"
    assert outcome.repair_attempts == 1
    assert fake.calls == 2
    assert "Ligne non conforme" in fake.prompts[1]  # le repair recoit l'erreur precise
    assert "Voici les trois questions" in fake.prompts[1]  # ... et le texte a corriger


def test_human_required_after_repair_fails_and_never_calls_a_third_time():
    llm, fake = client_with([INTRO, INTRO, GOOD])
    outcome = generate_research_questions(llm, TOPIC)
    assert outcome.status == "human_required"
    assert outcome.questions == []
    assert outcome.error
    assert fake.calls == 2  # un SEUL repair


def test_empty_topic_refused_without_any_llm_call():
    llm, fake = client_with([GOOD])
    with pytest.raises(ValueError, match="Sujet vide"):
        generate_research_questions(llm, "   ")
    assert fake.calls == 0


def test_budget_error_is_not_swallowed_as_a_format_problem():
    llm, fake = client_with([INTRO, GOOD], budget=Budget(max_calls=1))
    with pytest.raises(BudgetExceeded):
        generate_research_questions(llm, TOPIC)  # 1er appel OK, le repair depasse le budget
    assert fake.calls == 1


def test_trace_records_decisions(trace):
    llm, _ = client_with([INTRO, GOOD], trace=trace)
    generate_research_questions(llm, TOPIC, trace=trace)
    events = [e["event"] for e in trace.read() if e["agent"] == "protocol"]
    assert events == ["questions_invalid", "questions_repaired"]


# -- artefact -----------------------------------------------------------------------------
def test_outcome_roundtrip_through_store(store):
    llm, _ = client_with([GOOD])
    outcome = generate_research_questions(llm, TOPIC)
    store.write_artifact("questions", QuestionsOutcome, outcome)
    back = store.read_artifact("questions", QuestionsOutcome)
    assert back.status == "accepted" and back.questions[1].identifier == "RQ2"


def test_outcome_coherence_rules():
    with pytest.raises(ValueError, match="questions"):
        QuestionsOutcome(topic="t", status="accepted", text="x")
    with pytest.raises(ValueError, match="human_required"):
        QuestionsOutcome(topic="t", status="human_required", text="x")  # pas d'erreur expliquee