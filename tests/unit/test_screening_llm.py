import pytest

from slrresearch.core.errors import BudgetExceeded, ConfigError, LLMError
from slrresearch.core.llm import LLMClient
from slrresearch.core.schemas.protocol_plan import SearchConcept
from slrresearch.screening.llm_screener import (
    build_prompt, check_grounding, fingerprint, known_criteria_ids, load_prompt, parse_verdict, screen_paper,
)
from tests.fakes import FakeLLM
from tests.screening_helpers import TEMPLATE, make_paper, make_protocol, verdict_json

IDS = {"I1", "E1", "LANG"}


def client(*responses, trace=None):
    return LLMClient(FakeLLM(list(responses)), model_name="fake", role="screening", trace=trace),


class Boom:
    def __init__(self, exc): self.exc, self.calls = exc, 0
    def call(self, prompt):
        self.calls += 1
        raise self.exc


# -- prompt ----------------------------------------------------------------
def test_real_prompt_file_is_complete_and_fills_every_variable():
    template = load_prompt()
    prompt = build_prompt(template, make_paper(), make_protocol())
    assert "{{" not in prompt
    for expected in ("LLM et screening", "I1 :", "E1 :", "RQ1 :", "LLM for title screening", "2018", "LANG"):
        assert expected in prompt


def test_prompt_is_built_from_protocol_not_from_code():
    protocol = make_protocol(topic="Métformine et diabète")
    assert "Métformine et diabète" in build_prompt(TEMPLATE, make_paper(), protocol)


def test_paper_text_is_not_reinterpreted_as_a_template():
    paper = make_paper(title="Voir {{topic}} ici", abstract="Texte {{inclusion}} fin.")
    prompt = build_prompt(TEMPLATE, paper, make_protocol())
    assert "TITRE=Voir {{topic}} ici" in prompt and "RESUME=Texte {{inclusion}} fin." in prompt


def test_unknown_template_variable_is_a_config_error():
    with pytest.raises(ConfigError):
        build_prompt("{{inconnue}}", make_paper(), make_protocol())


def test_fingerprint_changes_with_criteria_topic_and_template():
    base = fingerprint(TEMPLATE, make_protocol())
    assert base == fingerprint(TEMPLATE, make_protocol())
    assert base != fingerprint(TEMPLATE + "x", make_protocol())
    assert base != fingerprint(TEMPLATE, make_protocol(topic="Autre"))
    assert base != fingerprint(TEMPLATE, make_protocol(year_from=2020))


# -- parse_verdict -----------------------------------------------------------
def test_parse_accepts_fenced_json_and_uppercase_decision():
    text = "Voici :\n```json\n" + verdict_json(decision="INCLUDE") + "\n```"
    v = parse_verdict(text, "W1", IDS)
    assert v.decision == "include" and v.paper_id == "W1"


def test_percentage_confidence_is_normalised():
    assert parse_verdict(verdict_json(confidence=85), "W1", IDS).confidence == pytest.approx(0.85)


def test_llm_cannot_choose_the_paper_id():
    text = verdict_json().replace("{", '{"paper_id": "HACK", ', 1)
    assert parse_verdict(text, "W1", IDS).paper_id == "W1"


@pytest.mark.parametrize("text,fragment", [
    ("", "vide"),
    ("pas de json", "JSON"),
    ('{"decision": "maybe", "confidence": 0.5, "reasons": ["x"]}', "decision"),
    ('{"decision": "include", "reasons": ["x"], "criteria": ["I1"], "evidence": "abc"}', "confidence"),
    (verdict_json(criteria=("I9",)), "inconnus"),
    (verdict_json(criteria=()), "critère"),
    (verdict_json(evidence=""), "preuve"),
    (verdict_json(confidence=7000), "confidence"),
    (verdict_json(reasons=()), "reasons"),
])
def test_parse_rejects_bad_answers(text, fragment):
    with pytest.raises(ValueError) as exc:
        parse_verdict(text, "W1", IDS)
    assert fragment.lower() in str(exc.value).lower()


def test_uncertain_needs_no_criteria_nor_evidence():
    v = parse_verdict(verdict_json(decision="uncertain", criteria=(), evidence="", confidence=0.3), "W1", IDS)
    assert v.decision == "uncertain"


def test_language_criterion_id_is_allowed():
    assert "LANG" in known_criteria_ids(make_protocol())


# -- grounding ---------------------------------------------------------------
def test_grounding_accepts_accent_and_case_variations():
    paper = make_paper(title="Évaluation des modèles", abstract="Nous testons un grand modèle de langage.")
    v = parse_verdict(verdict_json(evidence="GRAND MODELE DE LANGAGE"), "W1", IDS)
    check_grounding(paper, v)


def test_grounding_rejects_invented_or_too_short_evidence():
    paper = make_paper()
    with pytest.raises(ValueError):
        check_grounding(paper, parse_verdict(verdict_json(evidence="quantum teleportation of cats"), "W1", IDS))
    with pytest.raises(ValueError):
        check_grounding(paper, parse_verdict(verdict_json(evidence="llm"), "W1", IDS))


# -- screen_paper --------------------------------------------------------------
def test_screen_ok_uses_exactly_one_call():
    llm, = client(verdict_json())
    r = screen_paper(llm, make_paper(), make_protocol(), TEMPLATE)
    assert r.status == "ok" and r.verdict.decision == "include" and llm.model.calls == 1


def test_one_repair_then_ok():
    llm, = client("n'importe quoi", verdict_json())
    r = screen_paper(llm, make_paper(), make_protocol(), TEMPLATE)
    assert r.status == "ok" and llm.model.calls == 2
    assert "INVALIDE" in llm.model.prompts[1]


def test_two_bad_answers_give_invalid_format_and_no_third_call(trace):
    llm, = client("mauvais", "toujours mauvais", trace=trace)
    r = screen_paper(llm, make_paper(), make_protocol(), TEMPLATE, trace)
    assert r.status == "invalid_format" and r.verdict is None and llm.model.calls == 2
    raw = [e for e in trace.read() if e["event"] == "invalid_format"]
    assert raw and "toujours mauvais" in str(raw[0])


def test_ungrounded_evidence_downgrades_to_uncertain():
    llm, = client(verdict_json(evidence="quantum teleportation of cats"))
    r = screen_paper(llm, make_paper(), make_protocol(), TEMPLATE)
    assert r.status == "grounding_failed" and r.verdict.decision == "uncertain"
    assert r.verdict.confidence == 0 and r.verdict.evidence == ""


def test_llm_error_is_contained_per_paper():
    boom = Boom(LLMError("réseau"))
    r = screen_paper(boom, make_paper(), make_protocol(), TEMPLATE)
    assert r.status == "llm_error" and "réseau" in r.note and boom.calls == 1


def test_budget_exceeded_is_never_swallowed():
    with pytest.raises(BudgetExceeded):
        screen_paper(Boom(BudgetExceeded("fin")), make_paper(), make_protocol(), TEMPLATE)