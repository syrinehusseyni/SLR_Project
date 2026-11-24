import hashlib

import httpx
import pytest

from slrresearch.core.errors import BudgetExceeded
from slrresearch.core.schemas.documents import DocumentRecord
from slrresearch.download.agent import (
    DownloadSettings, download_paper, safe_filename, stored_file_ok,
)
from tests.download_helpers import config, make_http, make_paper, pdf_response
from tests.pdf_helpers import BODY, good_pdf, make_pdf

GOOD = good_pdf()
SCAN = make_pdf(["", ""])
OTHER = make_pdf(["Fine tuning of convolutional networks for tumour detection\n" + BODY])
HTML = b"<html><body>Please log in to read this article</body></html>"


def server(routes, default=None):
    """Faux serveur : routes = {fragment d'URL: réponse}. Sans correspondance : 404."""
    def handler(request):
        for fragment, response in routes.items():
            if fragment in str(request.url):
                return response() if callable(response) else response
        return default or httpx.Response(404)
    return handler


def run(tmp_path, routes, paper=None, settings=None, cfg=None, **kw):
    http, requests, sleeps = make_http(server(routes), **kw)
    record = download_paper(paper or make_paper(), http, cfg or config(), settings or DownloadSettings(), tmp_path / "documents")
    return record, requests, sleeps


def test_first_valid_link_wins_and_later_resolvers_are_not_called(tmp_path):
    paper = make_paper(pdf_url="https://repo.test/a.pdf", doi="10.1/x")
    record, requests, _ = run(tmp_path, {"repo.test": pdf_response(GOOD)}, paper)
    assert record.status == "downloaded" and record.has_text_layer
    assert record.pdf_path == "documents/W1.pdf" and record.sha256 == hashlib.sha256(GOOD).hexdigest()
    assert (tmp_path / "documents" / "W1.pdf").read_bytes() == GOOD
    assert not list((tmp_path / "documents").glob("*.part"))
    assert [a.resolver for a in record.attempts] == ["openalex_oa"]
    assert record.attempts[0].ok and record.attempts[0].validation == "ok" and record.attempts[0].http_status == 200
    assert len(requests) == 1  # ni Unpaywall ni Semantic Scholar ni Crossref


def test_html_page_then_unpaywall_pdf(tmp_path):
    paper = make_paper(pdf_url="https://pub.test/paywall", doi="10.1/x")
    routes = {
        "pub.test": httpx.Response(200, content=HTML),
        "api.unpaywall.org": httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://green.test/p.pdf"}}),
        "green.test": pdf_response(GOOD),
    }
    record, _, _ = run(tmp_path, routes, paper)
    assert record.status == "downloaded"
    assert [(a.resolver, a.validation, a.ok) for a in record.attempts] == [
        ("openalex_oa", "not_pdf", False), ("unpaywall", "ok", True)]


def test_wrong_article_is_rejected_then_next_candidate_is_tried(tmp_path):
    paper = make_paper(pdf_url="https://a.test/wrong.pdf", doi="10.48550/arxiv.2311.07918")
    record, _, _ = run(tmp_path, {"a.test": pdf_response(OTHER), "arxiv.org/pdf/2311.07918": pdf_response(GOOD)}, paper, cfg=config(email=""))
    assert record.status == "downloaded"
    assert [(a.resolver, a.validation) for a in record.attempts] == [
        ("openalex_oa", "wrong_title"), ("unpaywall", "skipped:no_contact_email"), ("europepmc", "lookup:no_record"),
        ("s2_open_access", "lookup:no_record"), ("arxiv", "ok")]


def test_scanned_pdf_ends_in_manual_required_not_downloaded(tmp_path):
    paper = make_paper(pdf_url="https://a.test/scan.pdf")
    record, _, _ = run(tmp_path, {"a.test": pdf_response(SCAN)}, paper)
    assert record.status == "manual_required" and record.pdf_path == "" and record.sha256 == ""
    assert record.attempts[0].validation == "no_text_layer"
    assert not (tmp_path / "documents").exists()  # rien n'est écrit sur le disque


def test_no_link_at_all_is_manual_required_with_empty_trace(tmp_path):
    record, requests, _ = run(tmp_path, {}, make_paper(doi=""))
    assert record.status == "manual_required" and record.attempts == [] and requests == []


def test_http_error_keeps_the_status(tmp_path):
    record, _, _ = run(tmp_path, {"pub.test": httpx.Response(403)}, make_paper(pdf_url="https://pub.test/a.pdf", doi=""))
    assert record.status == "manual_required"
    assert (record.attempts[0].validation, record.attempts[0].http_status) == ("http_error", 403)


def test_redirect_loop_is_traced_without_leaking_the_error_text(tmp_path):
    def loop():
        raise httpx.TooManyRedirects("boucle https://x.test?api_key=SECRET")
    record, _, _ = run(tmp_path, {"loop.test": loop}, make_paper(pdf_url="https://loop.test/a.pdf", doi=""))
    assert record.attempts[0].validation == "network_error:TooManyRedirects"
    assert "SECRET" not in record.model_dump_json()


def test_same_url_is_tried_only_once(tmp_path):
    paper = make_paper(pdf_url="https://a.test/x.pdf", doi="10.1/x")
    routes = {"a.test/x.pdf": httpx.Response(200, content=HTML),
              "api.unpaywall.org": httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://a.test/x.pdf"}})}
    record, requests, _ = run(tmp_path, routes, paper)
    assert [a.url for a in record.attempts if a.resolver in ("openalex_oa", "unpaywall")] == ["https://a.test/x.pdf"]
    assert sum(1 for r in requests if "a.test/x.pdf" in str(r.url)) == 1


def test_max_urls_per_resolver_is_a_setting(tmp_path):
    locations = [{"url_for_pdf": f"https://m.test/{i}.pdf"} for i in range(5)]
    routes = {"api.unpaywall.org": httpx.Response(200, json={"oa_locations": locations}), "m.test": httpx.Response(200, content=HTML)}
    paper = make_paper(doi="10.1/x")
    record, _, _ = run(tmp_path, routes, paper, DownloadSettings(max_urls_per_resolver=2))
    assert sum(1 for a in record.attempts if "m.test" in a.url) == 2


def test_max_bytes_is_a_setting(tmp_path):
    paper = make_paper(pdf_url="https://a.test/a.pdf")
    record, _, _ = run(tmp_path, {"a.test": pdf_response(GOOD)}, paper, DownloadSettings(max_bytes=200))
    assert record.status == "manual_required" and record.attempts[0].validation == "too_large"


def test_budget_exceeded_propagates_and_writes_nothing(tmp_path):
    paper = make_paper(pdf_url="https://a.test/a.pdf")
    http, _, _ = make_http(server({"a.test": pdf_response(GOOD)}), max_calls=0)
    with pytest.raises(BudgetExceeded):
        download_paper(paper, http, config(), DownloadSettings(), tmp_path / "documents")
    assert not (tmp_path / "documents").exists()


def test_arxiv_downloads_respect_the_arxiv_spacing(tmp_path):
    paper = make_paper(source="arxiv", pdf_url="https://arxiv.org/pdf/2311.07918", doi="")
    http, _, sleeps = make_http(server({"arxiv.org": pdf_response(GOOD)}), intervals={"arxiv": 3.0, "pdf": 0.0})
    http.limiter._last["arxiv"] = http.limiter.clock()  # un appel arXiv vient d'avoir lieu
    download_paper(paper, http, config(), DownloadSettings(), tmp_path / "documents")
    assert sleeps and sleeps[0] > 0


def test_record_survives_a_json_round_trip(tmp_path):
    record, _, _ = run(tmp_path, {"a.test": pdf_response(GOOD)}, make_paper(pdf_url="https://a.test/a.pdf"))
    assert DocumentRecord.model_validate_json(record.model_dump_json()) == record


def test_stored_file_ok_detects_missing_or_altered_files(tmp_path):
    record, _, _ = run(tmp_path, {"a.test": pdf_response(GOOD)}, make_paper(pdf_url="https://a.test/a.pdf"))
    assert stored_file_ok(record, tmp_path)
    (tmp_path / record.pdf_path).write_bytes(GOOD + b"x")
    assert not stored_file_ok(record, tmp_path)
    (tmp_path / record.pdf_path).unlink()
    assert not stored_file_ok(record, tmp_path)
    assert not stored_file_ok(DocumentRecord(paper_id="W1", status="manual_required"), tmp_path)


def test_safe_filename():
    assert safe_filename("doi-10-1093-jamia-ocaf063") == "doi-10-1093-jamia-ocaf063"
    assert safe_filename("../../evil id") == "evil_id"
    assert safe_filename("///") == "paper"


def test_pmc_html_page_then_europe_pmc_pdf(tmp_path):
    """Cas réel : Unpaywall donne une page PMC (pas un PDF) ; Europe PMC fournit le vrai PDF."""
    paper = make_paper(doi="10.2196/64682")
    epmc = {"resultList": {"result": [{"doi": "10.2196/64682", "pmcid": "PMC11922487", "fullTextUrlList": {"fullTextUrl": [
        {"availabilityCode": "OA", "documentStyle": "pdf", "site": "Europe_PMC", "url": "https://render.test/epmc.pdf"}]}}]}}
    routes = {
        "api.unpaywall.org": httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://blocked.test/pdf/"}}),
        "blocked.test": httpx.Response(200, content=b"<html>Preparing to download...</html>"),
        "ebi.ac.uk": httpx.Response(200, json=epmc),
        "render.test": pdf_response(GOOD),
    }
    record, _, _ = run(tmp_path, routes, paper)
    assert record.status == "downloaded"
    assert [(a.resolver, a.validation) for a in record.attempts] == [("unpaywall", "not_pdf"), ("europepmc", "ok")]


def test_landing_page_is_the_last_resort(tmp_path):
    paper = make_paper(doi="10.1/x")
    page = b'<html><meta name="citation_pdf_url" content="https://cdn.test/art.pdf"></html>'
    routes = {"doi.org": httpx.Response(200, content=page), "cdn.test": pdf_response(GOOD)}
    record, _, _ = run(tmp_path, routes, paper)
    assert record.status == "downloaded" and record.attempts[-1].resolver == "landing_page"
    assert [a.validation for a in record.attempts[:-1]] == ["lookup:no_record"] * (len(record.attempts) - 1)