import pytest

from slrresearch.core.schemas.protocol_plan import SearchConcept
from slrresearch.screening.rules import apply_rules
from slrresearch.screening.text import normalize_for_matching
from tests.screening_helpers import make_paper, make_protocol


def test_year_before_minimum_is_a_decisive_exclusion():
    v = apply_rules(make_paper(year=2015), make_protocol(year_from=2018))
    assert v.decision == "exclude" and v.decisive and "2015" in v.reasons[0]


def test_unknown_year_is_not_excluded_by_year():
    v = apply_rules(make_paper(year=None), make_protocol(year_from=2018))
    assert not v.decisive and v.decision == "include"


def test_no_year_limit_means_no_year_rule():
    assert apply_rules(make_paper(year=1990), make_protocol(year_from=None)).decision == "include"


def test_missing_abstract_is_uncertain_not_decisive():
    v = apply_rules(make_paper(abstract="   "), make_protocol())
    assert v.decision == "uncertain" and not v.decisive and "Résumé absent" in v.reasons[0]


def test_old_paper_without_abstract_is_excluded_by_year_first():
    assert apply_rules(make_paper(abstract="", year=2010), make_protocol()).decisive


def test_all_concepts_present_includes():
    v = apply_rules(make_paper(), make_protocol())
    assert v.decision == "include" and v.concepts_matched == ["C1", "C2"]


def test_no_concept_excludes_but_not_decisively():
    v = apply_rules(make_paper(title="Soil chemistry", abstract="We study soil."), make_protocol())
    assert v.decision == "exclude" and not v.decisive


def test_partial_coverage_is_uncertain():
    v = apply_rules(make_paper(title="LLM agents", abstract="A large language model agent."), make_protocol())
    assert v.decision == "uncertain" and v.concepts_matched == ["C1"] and "C2" in v.reasons[0]


def test_whole_words_only_and_plurals():
    protocol = make_protocol(concepts=[
        SearchConcept(id="C1", label="ia", terms=["AI", "machine learning"]),
        SearchConcept(id="C2", label="chaîne", terms=["supply chain", "logistics"]),
    ])
    # « ai » ne doit pas être trouvé dans « chain » ; « logistics » est présent
    no_ai = make_paper(title="Supply chain logistics", abstract="Chain management and logistics.")
    assert apply_rules(no_ai, protocol).decision == "uncertain"
    with_plural = make_paper(title="AIs and supply chains", abstract="Machine learnings and logistics.")
    assert apply_rules(with_plural, protocol).decision == "include"


def test_accents_and_case_are_neutralised():
    assert normalize_for_matching("Résumé  ÉTÉ") == "resume ete"
    protocol = make_protocol(concepts=[
        SearchConcept(id="C1", label="xx", terms=["résumé automatique", "summarisation"]),
        SearchConcept(id="C2", label="yy", terms=["évaluation", "benchmark"]),
    ])
    paper = make_paper(title="RESUME AUTOMATIQUE", abstract="Une EVALUATION sur corpus.")
    assert apply_rules(paper, protocol).decision == "include"


@pytest.mark.parametrize("topic,terms_a,terms_b,title,abstract", [
    ("Métformine et diabète", ["metformin", "biguanide"], ["type 2 diabetes", "glycemic control"],
     "Metformin in type 2 diabetes", "A trial of metformin for glycemic control."),
    ("Rénovation énergétique", ["heat pump", "geothermal"], ["residential building", "housing"],
     "Heat pumps in housing", "Field data on a heat pump in residential buildings."),
])
def test_rules_are_topic_independent(topic, terms_a, terms_b, title, abstract):
    protocol = make_protocol(topic=topic, concepts=[
        SearchConcept(id="C1", label="aa", terms=terms_a), SearchConcept(id="C2", label="bb", terms=terms_b)])
    assert apply_rules(make_paper(title=title, abstract=abstract), protocol).decision == "include"
    assert apply_rules(make_paper(title="Unrelated", abstract="Nothing relevant here."), protocol).decision == "exclude"