from slrresearch.core.schemas.papers import Paper
from slrresearch.search.dedup import deduplicate, normalize_title


def P(id, title, source, **kw):
    return Paper(id=id, title=title, source=source, **kw)


def test_normalize_title():
    assert normalize_title("  LLMs: A Survey!  ") == "llms a survey"


def test_same_doi_merged_and_fields_completed():
    a = P("W1", "Screening papers with large language models", "openalex", doi="10.1/a", abstract="", year=2024)
    b = P("doi-10-1-a", "Screening papers with large language models", "crossref", doi="10.1/a", abstract="Un résumé complet.", venue="J. X", authors=["A B"])
    res = deduplicate([a, b])
    assert len(res.papers) == 1 and res.duplicates_removed == 1
    p = res.papers[0]
    assert p.id == "W1" and p.source == "openalex"           # id OpenAlex prioritaire
    assert p.abstract == "Un résumé complet." and p.venue == "J. X" and p.authors == ["A B"] and p.year == 2024
    assert res.found_in == {"W1": ["crossref", "openalex"]}


def test_same_title_different_dois_merged_preprint_vs_published():
    a = P("arxiv-2401-1", "Zero-shot screening with large language models", "arxiv", doi="10.48550/arxiv.2401.1")
    b = P("doi-10-1-j", "Zero-Shot Screening with Large Language Models!", "crossref", doi="10.1/j")
    res = deduplicate([a, b])
    assert len(res.papers) == 1 and res.duplicates_removed == 1
    assert res.papers[0].source == "crossref" and res.papers[0].id == "doi-10-1-j"  # la version publiée prime


def test_short_titles_are_not_used_as_key():
    res = deduplicate([P("a", "Introduction", "crossref", doi="10.1/x"), P("b", "Introduction", "arxiv", doi="10.1/y")])
    assert len(res.papers) == 2 and res.duplicates_removed == 0


def test_same_id_repeated_across_queries():
    res = deduplicate([P("W9", "Some long enough title here", "openalex"), P("W9", "Some long enough title here", "openalex")])
    assert len(res.papers) == 1 and res.duplicates_removed == 1 and res.found_in == {"W9": ["openalex"]}


def test_id_without_openalex_derived_from_doi():
    res = deduplicate([P("s2-x", "Another sufficiently long title", "semanticscholar", doi="10.5/Z"),
                       P("arxiv-1", "Another sufficiently long title", "arxiv")])
    assert res.papers[0].id == "doi-10-5-z"


def test_longest_abstract_kept_and_open_access_any():
    res = deduplicate([P("a", "A long enough shared title text", "semanticscholar", doi="10.1/q", abstract="court", open_access=False),
                       P("b", "A long enough shared title text", "arxiv", doi="10.1/q", abstract="un résumé plus long", open_access=True)])
    assert res.papers[0].abstract == "un résumé plus long" and res.papers[0].open_access is True


def test_deterministic_and_order_preserved():
    recs = [P("W1", "First paper with a long title", "openalex"), P("W2", "Second paper with a long title", "openalex"),
            P("s2-1", "First paper with a long title", "semanticscholar")]
    r1, r2 = deduplicate(recs), deduplicate(list(recs))
    assert [p.id for p in r1.papers] == ["W1", "W2"] == [p.id for p in r2.papers]