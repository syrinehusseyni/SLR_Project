import pytest
from pydantic import ValidationError

from slrresearch.core.schemas.protocol_plan import SearchConcept
from slrresearch.protocol.search_plan import (
    arxiv_query, boolean_query, build_search_plan, keyword_variants, quote_term,
)


def concepts():
    return [
        SearchConcept(id="C1", label="task", terms=["systematic review", "literature review"]),
        SearchConcept(id="C2", label="method", terms=["large language model", "LLM", "GPT-4"]),
    ]


def test_quote_term():
    assert quote_term("LLM") == "LLM"
    assert quote_term("systematic review") == '"systematic review"'
    assert quote_term("GPT-4") == '"GPT-4"'


def test_boolean_query_exact():
    assert boolean_query(concepts()) == (
        '("systematic review" OR "literature review") AND ("large language model" OR LLM OR "GPT-4")'
    )


def test_arxiv_query_exact_and_year():
    assert arxiv_query(concepts()) == (
        '(all:"systematic review" OR all:"literature review") AND (all:"large language model" OR all:LLM OR all:"GPT-4")'
    )
    assert arxiv_query(concepts(), 2020).endswith("AND submittedDate:[202001010000 TO 209912312359]")


def test_keyword_variants_zip_terms_and_dedupe():
    assert keyword_variants(concepts()) == [
        "systematic review large language model",
        "literature review LLM",
        "systematic review GPT-4",
    ]
    one = [SearchConcept(id="C1", label="aa", terms=["alpha", "beta"]), SearchConcept(id="C2", label="bb", terms=["gamma", "delta"])]
    assert keyword_variants(one, max_variants=5) == ["alpha gamma", "beta delta"]  # s'arrête quand plus de termes


def test_plan_has_every_source_and_year_filters():
    plan = build_search_plan(concepts(), year_from=2018)
    sources = [q.source for q in plan.queries]
    assert set(sources) == {"openalex", "arxiv", "crossref", "semanticscholar"}
    by = {q.source: q for q in plan.queries}
    assert by["openalex"].query_param == "search.title_and_abstract"
    assert by["openalex"].params == {"filter": "from_publication_date:2018-01-01"}
    cross = [q for q in plan.queries if q.source == "crossref"]
    assert all(q.params == {"filter": "from-pub-date:2018-01-01"} for q in cross)
    s2 = [q for q in plan.queries if q.source == "semanticscholar"]
    assert all(q.params == {"year": "2018-"} for q in s2)
    assert " AND " not in cross[0].query and "(" not in cross[0].query  # pas de booléen pour Crossref


def test_plan_without_year_has_no_params():
    plan = build_search_plan(concepts())
    assert all(q.params == {} for q in plan.queries)


def test_plan_is_deterministic():
    assert build_search_plan(concepts(), 2020) == build_search_plan(concepts(), 2020)


@pytest.mark.parametrize("terms", [["a"], ["ok term", 'bad "quote"'], ["ok term", "(paren)"], ["ok term", "AND"], ["same", "SAME"]])
def test_concept_rejects_bad_terms(terms):
    with pytest.raises(ValidationError):
        SearchConcept(id="C1", label="xx", terms=terms)


def test_concept_dedupes_and_normalizes_spaces():
    c = SearchConcept(id="C1", label=" a  b ", terms=["deep   learning", "Deep Learning", "CNN"])
    assert c.terms == ["deep learning", "CNN"] and c.label == "a b"