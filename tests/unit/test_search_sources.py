import pytest

from slrresearch.search.http import SearchError
from slrresearch.search.sources import (
    clean_jats, parse_arxiv, parse_crossref, parse_openalex, parse_semanticscholar, reconstruct_abstract, slug,
)

OPENALEX = {"results": [
    {"id": "https://openalex.org/W111", "doi": "https://doi.org/10.1000/ABC.1", "title": "LLMs for citation screening",
     "publication_year": 2024,
     "abstract_inverted_index": {"We": [0], "test": [1, 3], "LLM": [2], "screening": [4]},
     "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": {"display_name": "Alan Turing"}}],
     "primary_location": {"landing_page_url": "https://example.org/p1", "source": {"display_name": "J. of Tests"}, "pdf_url": None},
     "best_oa_location": {"pdf_url": "https://example.org/p1.pdf"}, "open_access": {"is_oa": True}},
    {"id": "https://openalex.org/W222", "doi": None, "title": None, "display_name": "Fallback title here",
     "publication_year": None, "abstract_inverted_index": None, "authorships": [], "primary_location": None,
     "best_oa_location": None, "open_access": {"is_oa": False}},
    {"id": "https://openalex.org/W333", "title": "", "display_name": ""},  # sans titre : écartée
]}

CROSSREF = {"message": {"items": [
    {"DOI": "10.1000/XYZ.2", "title": ["Screening with GPT-4"], "abstract": "<jats:p><jats:title>Abstract</jats:title>We study <jats:italic>GPT-4</jats:italic> &amp; recall.</jats:p>",
     "issued": {"date-parts": [[2023, 5, 1]]}, "author": [{"given": "Grace", "family": "Hopper"}, {"name": "ACME Lab"}],
     "container-title": ["Journal X"], "URL": "https://doi.org/10.1000/xyz.2"},
    {"DOI": "10.1000/NOTITLE", "title": [], "issued": {"date-parts": [[None]]}},
    {"DOI": "10.1000/REVIEW", "title": ["Review for \"Screening with GPT-4\""], "type": "peer-review"},
]}}

SEMANTIC = {"total": 2, "data": [
    {"paperId": "abc123", "title": "Prompting for screening", "abstract": None, "year": 2022,
     "authors": [{"authorId": "1", "name": "Joan Clarke"}], "venue": "ACL", "externalIds": {"DOI": "10.1000/S2.3", "ArXiv": "2201.00001"},
     "openAccessPdf": {"url": "https://example.org/s2.pdf"}, "isOpenAccess": True, "url": "https://www.semanticscholar.org/paper/abc123"},
    {"paperId": "def456", "title": None},
]}

ARXIV = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <published>2024-01-03T10:00:00Z</published>
    <title>Zero-shot   screening
      with LLMs</title>
    <summary>  We evaluate zero-shot screening.  </summary>
    <author><name>Katherine Johnson</name></author>
    <arxiv:doi>10.1000/arx.4</arxiv:doi>
  </entry>
  <entry><id>http://arxiv.org/abs/hep-th/9711200v1</id><published>1997-11-27T00:00:00Z</published>
    <title>Old style id paper</title><summary>x</summary></entry>
</feed>"""

ARXIV_ERROR = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad</id>
<title>Error</title><summary>incorrect id format</summary></entry></feed>"""


def test_reconstruct_abstract_orders_words():
    assert reconstruct_abstract({"We": [0], "test": [1, 3], "LLM": [2], "screening": [4]}) == "We test LLM test screening"
    assert reconstruct_abstract(None) == "" and reconstruct_abstract({"a": "bad"}) == ""


def test_parse_openalex():
    res = parse_openalex(OPENALEX)
    assert res.skipped == 1 and len(res.papers) == 2
    p = res.papers[0]
    assert p.id == "W111" and p.doi == "10.1000/abc.1" and p.year == 2024 and p.source == "openalex"
    assert p.abstract == "We test LLM test screening"
    assert p.authors == ["Ada Lovelace", "Alan Turing"] and p.venue == "J. of Tests"
    assert p.open_access is True and p.pdf_url == "https://example.org/p1.pdf" and p.url == "https://example.org/p1"
    q = res.papers[1]
    assert q.title == "Fallback title here" and q.abstract == "" and q.year is None and q.doi == ""


def test_parse_crossref():
    res = parse_crossref(CROSSREF)
    assert res.skipped == 2 and len(res.papers) == 1  # sans titre + compte rendu d'évaluation (peer-review)
    p = res.papers[0]
    assert p.id == "doi-10-1000-xyz-2" and p.doi == "10.1000/xyz.2" and p.year == 2023
    assert p.abstract == "We study GPT-4 & recall." and p.authors == ["Grace Hopper", "ACME Lab"]
    assert p.venue == "Journal X" and p.open_access is False and p.pdf_url == ""


def test_crossref_keeps_articles_and_proceedings_but_drops_noise_types():
    items = [{"DOI": f"10.1/{t}", "title": [f"A sufficiently titled {t}"], "type": t}
             for t in ("journal-article", "proceedings-article", "posted-content", "peer-review", "dataset", "grant")]
    res = parse_crossref({"message": {"items": items}})
    assert [p.doi for p in res.papers] == ["10.1/journal-article", "10.1/proceedings-article", "10.1/posted-content"]
    assert res.skipped == 3


def test_clean_jats():
    assert clean_jats("<p>Abstract: Hello <b>world</b></p>") == "Hello world"
    assert clean_jats(None) == ""


def test_parse_semanticscholar():
    res = parse_semanticscholar(SEMANTIC)
    assert res.skipped == 1 and len(res.papers) == 1
    p = res.papers[0]
    assert p.id == "s2-abc123" and p.doi == "10.1000/s2.3" and p.abstract == "" and p.open_access is True
    assert p.pdf_url == "https://example.org/s2.pdf" and p.authors == ["Joan Clarke"]


def test_parse_arxiv():
    res = parse_arxiv(ARXIV)
    assert len(res.papers) == 2
    p = res.papers[0]
    assert p.id == "arxiv-2401-01234" and p.title == "Zero-shot screening with LLMs"
    assert p.abstract == "We evaluate zero-shot screening." and p.year == 2024 and p.doi == "10.1000/arx.4"
    assert p.pdf_url == "https://arxiv.org/pdf/2401.01234" and p.open_access is True
    assert res.papers[1].id == "arxiv-hep-th-9711200"


def test_arxiv_error_entry_and_bad_xml_raise():
    with pytest.raises(SearchError):
        parse_arxiv(ARXIV_ERROR)
    with pytest.raises(SearchError):
        parse_arxiv("<<pas du xml")


@pytest.mark.parametrize("parser,payload", [(parse_openalex, {}), (parse_openalex, []), (parse_crossref, {"message": 1}),
                                            (parse_semanticscholar, {"data": "x"})])
def test_unexpected_payload_raises(parser, payload):
    with pytest.raises(SearchError):
        parser(payload)


def test_slug():
    assert slug("10.1000/ABC.1") == "10-1000-ABC-1"