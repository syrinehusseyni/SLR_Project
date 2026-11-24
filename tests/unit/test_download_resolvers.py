import httpx
import pytest

from slrresearch.core.errors import BudgetExceeded
from slrresearch.download import resolvers as R
from tests.download_helpers import config, make_http, make_paper


def test_clean_doi_removes_prefixes():
    assert R.clean_doi("https://doi.org/10.1/abc") == "10.1/abc"
    assert R.clean_doi(" doi:10.1/abc ") == "10.1/abc"
    assert R.clean_doi("") == ""


def test_known_link_is_labelled_by_source():
    http, requests, _ = make_http(lambda r: httpx.Response(500))
    res = R.resolve_known_link(make_paper(source="openalex", pdf_url="https://x.test/a.pdf"), http, config())
    assert (res.resolver, res.urls) == ("openalex_oa", ["https://x.test/a.pdf"])
    assert R.resolve_known_link(make_paper(source="arxiv", pdf_url="https://arxiv.org/pdf/1"), http, config()).resolver == "arxiv"
    assert R.resolve_known_link(make_paper(source="semanticscholar", pdf_url="u"), http, config()).resolver == "s2_open_access"
    assert R.resolve_known_link(make_paper(pdf_url=""), http, config()).urls == []
    assert requests == []  # aucun appel réseau


# ---------------------------------------------------------------- Unpaywall
def test_parse_unpaywall_best_then_others_without_duplicates():
    payload = {"best_oa_location": {"url_for_pdf": "https://a/1.pdf"},
               "oa_locations": [{"url_for_pdf": "https://a/1.pdf"}, {"url_for_pdf": None}, {"url_for_pdf": "https://b/2.pdf"}, "bizarre"]}
    assert R.parse_unpaywall(payload) == ["https://a/1.pdf", "https://b/2.pdf"]
    assert R.parse_unpaywall({"best_oa_location": None, "oa_locations": None}) == []
    assert R.parse_unpaywall([]) == []


def test_unpaywall_request_has_email_and_contact_user_agent():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://a/1.pdf"}}))
    res = R.resolve_unpaywall(make_paper(doi="https://doi.org/10.1000/xyz(1)"), http, config("me@lab.org"))
    assert res.urls == ["https://a/1.pdf"] and res.note is None
    request = requests[0]
    assert request.url.host == "api.unpaywall.org" and request.url.path.startswith("/v2/10.1000/xyz")
    assert request.url.params["email"] == "me@lab.org"
    assert "mailto:me@lab.org" in request.headers["user-agent"]


def test_unpaywall_without_email_is_skipped_without_network():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={}))
    res = R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config(email=""))
    assert res.urls == [] and res.note.validation == "skipped:no_contact_email" and requests == []


def test_unpaywall_without_doi_does_nothing():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={}))
    res = R.resolve_unpaywall(make_paper(doi=""), http, config())
    assert res.urls == [] and res.note is None and requests == []


@pytest.mark.parametrize("status,expected", [(404, "lookup:no_record"), (422, "lookup:HTTP 422"), (429, "lookup:HTTP 429")])
def test_unpaywall_failures_are_traced_not_raised(status, expected):
    http, _, _ = make_http(lambda r: httpx.Response(status))
    res = R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config())
    assert res.urls == [] and res.note.validation == expected and res.note.http_status == status and not res.note.ok


def test_unpaywall_record_without_pdf_link_is_traced():
    http, _, _ = make_http(lambda r: httpx.Response(200, json={"is_oa": False, "best_oa_location": None}))
    res = R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config())
    assert res.urls == [] and res.note.validation == "lookup:no_pdf_link"


def test_lookup_invalid_json_and_redirect_loop_are_traced():
    http, _, _ = make_http(lambda r: httpx.Response(200, content=b"<html>pas du json</html>"))
    assert R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config()).note.validation == "lookup:invalid_json"

    def loop(request):
        raise httpx.TooManyRedirects("boucle https://x.test?api_key=SECRET")
    http, _, _ = make_http(loop)
    note = R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config()).note
    assert note.validation == "lookup:TooManyRedirects" and "SECRET" not in note.model_dump_json()


def test_budget_exceeded_is_never_swallowed():
    http, _, _ = make_http(lambda r: httpx.Response(200, json={}), max_calls=0)
    with pytest.raises(BudgetExceeded):
        R.resolve_unpaywall(make_paper(doi="10.1/x"), http, config())


# ---------------------------------------------------------------- Semantic Scholar
def test_semantic_identifier():
    assert R.semantic_identifier(make_paper(doi="10.1/x")) == "DOI:10.1/x"
    assert R.semantic_identifier(make_paper(id="s2-abc123", doi="")) == "abc123"
    assert R.semantic_identifier(make_paper(id="W9", doi="")) == ""


def test_semantic_lookup_uses_key_and_reads_open_access_pdf():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={"openAccessPdf": {"url": "https://s/p.pdf", "status": "GREEN"}}))
    res = R.resolve_semantic_scholar(make_paper(doi="10.1/x"), http, config(s2_key="CLE"))
    assert res.resolver == "s2_open_access" and res.urls == ["https://s/p.pdf"]
    assert requests[0].headers["x-api-key"] == "CLE" and requests[0].url.params["fields"] == "openAccessPdf"
    assert requests[0].url.path.endswith("/paper/DOI:10.1/x")


def test_semantic_without_open_access_or_identifier():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={"openAccessPdf": None}))
    assert R.resolve_semantic_scholar(make_paper(doi="10.1/x"), http, config()).note.validation == "lookup:no_pdf_link"
    n = len(requests)
    assert R.resolve_semantic_scholar(make_paper(id="W9", doi=""), http, config()).urls == [] and len(requests) == n


def test_semantic_429_is_traced():
    http, _, _ = make_http(lambda r: httpx.Response(429))
    res = R.resolve_semantic_scholar(make_paper(doi="10.1/x"), http, config())
    assert res.urls == [] and res.note.validation == "lookup:HTTP 429"


# ---------------------------------------------------------------- arXiv
def test_arxiv_id_from_doi_or_url_without_version():
    assert R.arxiv_id(make_paper(doi="10.48550/arxiv.2311.07918")) == "2311.07918"
    assert R.arxiv_id(make_paper(doi="10.48550/arXiv.2311.07918v2")) == "2311.07918"
    assert R.arxiv_id(make_paper(doi="", url="http://arxiv.org/abs/2405.14445v3")) == "2405.14445"
    assert R.arxiv_id(make_paper(doi="", url="https://arxiv.org/pdf/2405.14445.pdf")) == "2405.14445"
    assert R.arxiv_id(make_paper(doi="10.1016/j.jbi.2025.104860", url="https://doi.org/10.1016/j.jbi.2025.104860")) == ""


def test_arxiv_resolver_builds_pdf_url_without_network():
    http, requests, _ = make_http(lambda r: httpx.Response(500))
    assert R.resolve_arxiv(make_paper(doi="10.48550/arxiv.2311.07918"), http, config()).urls == ["https://arxiv.org/pdf/2311.07918"]
    assert R.resolve_arxiv(make_paper(doi="10.1/x"), http, config()).urls == [] and requests == []


# ---------------------------------------------------------------- Crossref
def test_parse_crossref_links_keeps_only_pdf_links():
    payload = {"message": {"link": [
        {"URL": "https://p/a.pdf", "content-type": "application/pdf"},
        {"URL": "https://p/b.xml", "content-type": "text/xml"},
        {"URL": "https://p/c.pdf?token=1", "content-type": "unspecified"},
        {"URL": "https://p/a.pdf", "content-type": "application/pdf"},
    ]}}
    assert R.parse_crossref_links(payload) == ["https://p/a.pdf", "https://p/c.pdf?token=1"]
    assert R.parse_crossref_links({"message": {}}) == [] and R.parse_crossref_links(None) == []


def test_crossref_resolver_uses_mailto_and_traces_no_link():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json={"message": {"link": []}}))
    res = R.resolve_crossref(make_paper(doi="10.1/x"), http, config("me@lab.org"))
    assert res.urls == [] and res.note.validation == "lookup:no_pdf_link"
    assert requests[0].url.params["mailto"] == "me@lab.org" and requests[0].url.path == "/works/10.1/x"


def test_resolver_order_follows_the_specification():
    assert [f.__name__ for f in R.RESOLVERS] == [
        "resolve_known_link", "resolve_unpaywall", "resolve_europepmc", "resolve_semantic_scholar", "resolve_arxiv",
        "resolve_crossref", "resolve_landing_page"]


def test_download_intervals_keep_search_limits_and_add_pdf():
    assert R.DOWNLOAD_INTERVALS["arxiv"] == 3.0 and R.DOWNLOAD_INTERVALS["pdf"] > 0
    assert "unpaywall" in R.DOWNLOAD_INTERVALS and "europepmc" in R.DOWNLOAD_INTERVALS


# ---------------------------------------------------------------- Europe PMC
EPMC_RECORD = {"resultList": {"result": [{
    "doi": "10.2196/64682", "pmcid": "PMC11922487", "isOpenAccess": "Y",
    "fullTextUrlList": {"fullTextUrl": [
        {"availability": "Open access", "availabilityCode": "OA", "documentStyle": "pdf", "site": "Europe_PMC",
         "url": "https://europepmc.org/articles/PMC11922487?pdf=render"},
        {"availability": "Open access", "availabilityCode": "OA", "documentStyle": "html", "site": "Europe_PMC",
         "url": "https://europepmc.org/articles/PMC11922487"},
        {"availability": "Subscription required", "availabilityCode": "S", "documentStyle": "pdf", "site": "DOI",
         "url": "https://doi.org/payant.pdf"},
        {"availability": "Free", "availabilityCode": "F", "documentStyle": "pdf", "site": "Publisher",
         "url": "https://pub.test/libre.pdf"},
    ]}}]}}


def test_parse_europepmc_keeps_free_pdfs_then_the_pmc_render_address():
    assert R.parse_europepmc(EPMC_RECORD, "10.2196/64682") == [
        "https://europepmc.org/articles/PMC11922487?pdf=render", "https://pub.test/libre.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC11922487&blobtype=pdf"]


def test_parse_europepmc_refuses_a_record_for_another_doi_or_no_record():
    assert R.parse_europepmc(EPMC_RECORD, "10.9999/autre") == []
    assert R.parse_europepmc({"resultList": {"result": []}}, "10.2196/64682") == []
    assert R.parse_europepmc(None, "10.2196/64682") == [] and R.parse_europepmc({}, "x") == []


def test_parse_europepmc_ignores_a_malformed_pmcid():
    record = {"resultList": {"result": [{"doi": "10.1/x", "pmcid": "../etc", "fullTextUrlList": {"fullTextUrl": []}}]}}
    assert R.parse_europepmc(record, "10.1/x") == []


def test_europepmc_request_is_a_quoted_doi_query_without_key_or_email():
    http, requests, _ = make_http(lambda r: httpx.Response(200, json=EPMC_RECORD))
    res = R.resolve_europepmc(make_paper(doi="https://doi.org/10.2196/64682"), http, config(email=""))
    assert res.resolver == "europepmc" and len(res.urls) == 3 and res.note is None
    params = requests[0].url.params
    assert requests[0].url.host == "www.ebi.ac.uk" and params["query"] == 'DOI:"10.2196/64682"'
    assert (params["format"], params["resultType"], params["pageSize"]) == ("json", "core", "1")


def test_europepmc_without_doi_does_nothing_and_failures_are_traced():
    http, requests, _ = make_http(lambda r: httpx.Response(500))
    assert R.resolve_europepmc(make_paper(doi=""), http, config()).urls == [] and requests == []
    res = R.resolve_europepmc(make_paper(doi="10.1/x"), http, config())
    assert res.urls == [] and res.note.validation == "lookup:HTTP 500"


def test_europepmc_unknown_record_is_traced_as_no_pdf_link():
    http, _, _ = make_http(lambda r: httpx.Response(200, json={"resultList": {"result": []}}))
    assert R.resolve_europepmc(make_paper(doi="10.1/x"), http, config()).note.validation == "lookup:no_pdf_link"


# ---------------------------------------------------------------- page de l'article (citation_pdf_url)
PAGE = """<html><head>
<meta name="citation_title" content="Un titre">
<meta content="/article/12/pdf?a=1&amp;b=2" name="citation_pdf_url">
<META NAME='citation_pdf_url' CONTENT='https://cdn.test/second.pdf'>
<meta name="citation_pdf_url" content="javascript:alert(1)">
<meta name="citation_pdf_url" content="">
</head><body>meta name="citation_pdf_url" content="https://dans-le-corps.test/x.pdf"</body></html>"""


def test_parse_citation_pdf_urls_handles_attribute_order_case_relative_urls_and_entities():
    assert R.parse_citation_pdf_urls(PAGE, "https://pub.test/article/12") == [
        "https://pub.test/article/12/pdf?a=1&b=2", "https://cdn.test/second.pdf"]
    assert R.parse_citation_pdf_urls("<html>rien</html>", "https://x.test") == []


def test_landing_page_follows_doi_and_reads_the_meta_tag():
    def handler(request):
        return httpx.Response(200, content=PAGE.encode(), headers={"content-type": "text/html"})
    http, requests, _ = make_http(handler)
    res = R.resolve_landing_page(make_paper(doi="10.2196/64682"), http, config())
    assert res.resolver == "landing_page" and res.urls[1] == "https://cdn.test/second.pdf" and res.note is None
    assert requests[0].url.host == "doi.org" and requests[0].url.path == "/10.2196/64682"
    assert "text/html" in requests[0].headers["accept"]


def test_landing_page_without_tag_without_doi_or_blocked_is_traced_not_raised():
    http, requests, _ = make_http(lambda r: httpx.Response(200, content=b"<html>Preparing to download...</html>"))
    assert R.resolve_landing_page(make_paper(doi="10.1/x"), http, config()).note.validation == "lookup:no_pdf_link"
    n = len(requests)
    assert R.resolve_landing_page(make_paper(doi=""), http, config()).urls == [] and len(requests) == n
    blocked, _, _ = make_http(lambda r: httpx.Response(403))
    note = R.resolve_landing_page(make_paper(doi="10.1/x"), blocked, config()).note
    assert (note.validation, note.http_status) == ("lookup:HTTP 403", 403)


def test_landing_page_that_is_directly_a_pdf_is_returned_as_candidate():
    http, _, _ = make_http(lambda r: httpx.Response(200, content=b"%PDF-1.4 ..."))
    assert R.resolve_landing_page(make_paper(doi="10.1/x"), http, config()).urls == ["https://doi.org/10.1/x"]


def test_parse_citation_pdf_urls_only_reads_the_beginning_of_a_huge_page():
    tag = '<meta name="citation_pdf_url" content="https://x.test/a.pdf">'
    assert R.parse_citation_pdf_urls(tag, "https://x.test") == ["https://x.test/a.pdf"]
    assert R.parse_citation_pdf_urls("x" * R.MAX_LANDING_CHARS + tag, "https://x.test") == []