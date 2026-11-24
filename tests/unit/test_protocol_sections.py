import pytest

from slrresearch.core.errors import LLMError
from slrresearch.core.schemas.protocol import ResearchQuestion
from slrresearch.protocol.common import normalize_text
from slrresearch.protocol.concepts import generate_concepts, parse_concepts
from slrresearch.protocol.criteria import generate_criteria, parse_criteria
from slrresearch.protocol.extraction_schema import generate_facets, parse_facets

QUESTIONS = [
    ResearchQuestion(identifier="RQ1", question="Quelle précision obtiennent les LLM au screening ?"),
    ResearchQuestion(identifier="RQ2", question="Quelles limites sont rapportées dans les études ?"),
]
GOOD_CONCEPTS = "task | title screening; abstract screening; study selection\nmethod | large language model; LLM; GPT-4\n"
GOOD_CRITERIA = (
    "INCLUDE | L'étude évalue un LLM pour le screening\nINCLUDE | L'étude rapporte au moins une métrique\n"
    "EXCLUDE | L'article n'est pas évalué empiriquement\nEXCLUDE | L'article est une simple opinion\n"
)
GOOD_FACETS = (
    "RQ1 | Métrique | Mesure de performance rapportée | oui\nRQ1 | Jeu de données | Corpus utilisé | non\n"
    "RQ2 | Limites | Limites citées par les auteurs | oui\n"
)


class Client:
    """Client minimal exposant .call(prompt).text, comme LLMClient."""
    def __init__(self, texts):
        self.texts, self.calls, self.prompts = list(texts), 0, []

    def call(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1

        class R:
            pass

        r = R()
        r.text = self.texts.pop(0)
        return r


def test_normalize_text_typography():
    assert normalize_text("a\u2011b \u2019 \u201cx\u201d\u00a0y") == "a-b ' \"x\" y"


def test_parse_concepts_ok_and_ids_assigned():
    cs = parse_concepts(GOOD_CONCEPTS)
    assert [c.id for c in cs] == ["C1", "C2"]
    assert cs[1].terms == ["large language model", "LLM", "GPT-4"]


def test_parse_concepts_accepts_markdown_bullets_and_typography():
    cs = parse_concepts("- task | title\u2011screening; abstract screening\n* method | LLM; large language model\n")
    assert cs[0].terms[0] == "title-screening"


@pytest.mark.parametrize("bad", ["", "juste du texte", "a | b", "a | x; y | z", "task | onlyone", "task | (x); y", "t | aa; bb"])
def test_parse_concepts_rejects(bad):
    with pytest.raises(ValueError):
        parse_concepts(bad)


def test_parse_concepts_rejects_too_many_and_duplicate_labels():
    five = "".join(f"l{i} | aa{i}; bb{i}\n" for i in range(5))
    with pytest.raises(ValueError):
        parse_concepts(five)
    with pytest.raises(ValueError):
        parse_concepts("same | aa; bb\nSame | cc; dd\n")


def test_parse_criteria_ok():
    c = parse_criteria(GOOD_CRITERIA, year_from=2018)
    assert [x.id for x in c.inclusion] == ["I1", "I2"] and [x.id for x in c.exclusion] == ["E1", "E2"]
    assert c.year_from == 2018 and c.languages == ["en"]


@pytest.mark.parametrize("bad", ["", "INCLUDE | court", "MAYBE | une phrase assez longue", GOOD_CRITERIA.split("EXCLUDE")[0], "INCLUDE x"])
def test_parse_criteria_rejects(bad):
    with pytest.raises(ValueError):
        parse_criteria(bad)


def test_parse_facets_ok():
    fs = parse_facets(GOOD_FACETS, ["RQ1", "RQ2"])
    assert [f.id for f in fs] == ["F1", "F2", "F3"]
    assert [f.blocking for f in fs] == [True, False, True]


@pytest.mark.parametrize("bad", [
    "", "RQ1 | a | desc longue | oui",  # RQ2 sans facette
    GOOD_FACETS + "RQ9 | x | desc longue | oui\n",
    "RQ1 | a | desc longue | peut-être\nRQ2 | b | desc longue | oui\n",
    "RQ1 | a | desc longue\nRQ2 | b | desc longue | oui\n",
])
def test_parse_facets_rejects(bad):
    with pytest.raises(ValueError):
        parse_facets(bad, ["RQ1", "RQ2"])


def test_concepts_accepted_no_repair():
    llm = Client([GOOD_CONCEPTS])
    res = generate_concepts(llm, "sujet", QUESTIONS)
    assert (res.status, llm.calls, res.repair_attempts) == ("accepted", 1, 0)


def test_concepts_exactly_one_repair_then_ok():
    llm = Client(["n'importe quoi", GOOD_CONCEPTS])
    res = generate_concepts(llm, "sujet", QUESTIONS)
    assert (res.status, llm.calls, res.repair_attempts) == ("repaired", 2, 1)
    assert "n'importe quoi" in llm.prompts[1]  # le repair voit la réponse fautive


def test_concepts_human_required_after_single_repair():
    llm = Client(["mauvais", "encore mauvais", "ne doit jamais être consommé"])
    res = generate_concepts(llm, "sujet", QUESTIONS)
    assert (res.status, llm.calls) == ("human_required", 2)
    assert res.error and res.items == []


def test_criteria_and_facets_repair_paths():
    c = generate_criteria(Client(["x", GOOD_CRITERIA]), "s", QUESTIONS, year_from=2020)
    assert c.status == "repaired" and c.items[0].year_from == 2020
    f = generate_facets(Client(["x", GOOD_FACETS]), "s", QUESTIONS)
    assert f.status == "repaired" and len(f.items) == 3


def test_trace_records_sections(trace):
    generate_concepts(Client(["x", GOOD_CONCEPTS]), "s", QUESTIONS, trace=trace)
    events = [e["event"] for e in trace.read()]
    assert events == ["concepts_invalid", "concepts_repaired"]


def test_llm_error_propagates():
    class Boom:
        def call(self, prompt):
            raise LLMError("réseau")

    with pytest.raises(LLMError):
        generate_concepts(Boom(), "s", QUESTIONS)



def test_clean_lines_strips_markdown_and_numbering():
    from slrresearch.protocol.common import clean_lines
    assert clean_lines("**task** | a b; c d\n1. method | x y; z w\n- `m` | q r; s t\n") == [
        "task | a b; c d", "method | x y; z w", "m | q r; s t"]


def test_parse_concepts_accepts_bold_and_numbered_lines():
    cs = parse_concepts("1. **task** | title screening; abstract screening\n2. **method** | LLM; large language model\n")
    assert [c.label for c in cs] == ["task", "method"]


def test_failed_section_logs_raw_text_in_trace(trace):
    generate_concepts(Client(["**Concept 1 : X**", "toujours pas"]), "s", QUESTIONS, trace=trace)
    events = trace.read()
    assert events[0]["event"] == "concepts_invalid" and events[0]["data"]["text"] == "**Concept 1 : X**"
    assert events[-1]["event"] == "concepts_human_required" and events[-1]["data"]["text"] == "toujours pas"