import httpx
import pytest

from slrresearch.core.schemas.documents import DocumentsOutcome, DocumentsProgress
from slrresearch.download.agent import DownloadSettings
from slrresearch.download.stage import (
    DownloadStage, check_manual_pdf, looks_unreachable, manual_pdf_path, manual_pdf_problem,
)
from tests.download_helpers import config
from tests.download_stage_helpers import paper, setup_run
from tests.pdf_helpers import BODY, good_pdf, make_pdf

GOOD = good_pdf()
SCAN = make_pdf(["", ""])
OTHER = make_pdf(["Fine tuning of convolutional networks for tumour detection\n" + BODY])
HTML = b"<html>Please log in</html>"


class Server:
    """Faux serveur : routes {fragment d'URL: octets}. Compte les requêtes. Sans correspondance : 404."""

    def __init__(self, routes):
        self.routes, self.requests = dict(routes), []

    def __call__(self, request):
        self.requests.append(str(request.url))
        for fragment, body in self.routes.items():
            if fragment in str(request.url):
                if isinstance(body, BaseException):
                    raise body
                return httpx.Response(200, content=body, headers={"content-type": "application/pdf"})
        return httpx.Response(404)


def make_stage(server, **kw):
    kw.setdefault("pause", 0.0)
    kw.setdefault("sleep", lambda s: None)
    return DownloadStage(config(email=""), transport=httpx.MockTransport(server), **kw)


def three_papers():
    return [paper("A", pdf_url="https://good.test/a.pdf"),
            paper("B", pdf_url="https://paywall.test/b.pdf"),
            paper("C", pdf_url="https://scan.test/c.pdf")]


ROUTES = {"good.test": GOOD, "paywall.test": HTML, "scan.test": SCAN}


def outcome(store):
    return store.read_artifact("documents", DocumentsOutcome)


def statuses(store):
    return {r.paper_id: r.status for r in outcome(store).records}


# ---------------------------------------------------------------- conditions de départ
def test_requires_screening_and_search_artifacts(store):
    from slrresearch.core.config import Settings
    from slrresearch.core.contract import build_context
    from tests.fakes import FakeLLM
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: FakeLLM([]))
    assert "screening" in make_stage(Server({})).run(ctx).message
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    store.artifact_path("search").unlink()
    assert "search" in make_stage(Server({})).run(ctx).message


def test_refuses_while_g2_points_are_pending(store):
    ctx = setup_run(store, three_papers(), ["include", "pending", "exclude"])
    server = Server(ROUTES)
    result = make_stage(server).run(ctx)
    assert result.status == "failed" and "G2" in result.message and server.requests == []


def test_refuses_when_nothing_is_included(store):
    ctx = setup_run(store, three_papers(), ["exclude"] * 3)
    assert "aucun papier inclus" in make_stage(Server({})).run(ctx).message


# ---------------------------------------------------------------- run complet
def test_full_run_downloads_included_only_and_opens_g3(store):
    papers = three_papers() + [paper("X", pdf_url="https://good.test/x.pdf")]
    ctx = setup_run(store, papers, ["include", "include", "include", "exclude"])
    server = Server(ROUTES)
    result = make_stage(server).run(ctx)
    assert result.status == "waiting_human" and result.artifact == "documents"
    assert {i.id for i in result.review_items} == {"G3:paper:B", "G3:paper:C"}
    assert all("Aucun PDF valide" in i.reason for i in result.review_items)
    assert not any("x.pdf" in url for url in server.requests)  # le papier exclu n'est jamais demandé
    assert statuses(store) == {"A": "downloaded", "B": "manual_required", "C": "manual_required"}
    out = outcome(store)
    assert out.summary.downloaded == 1 and out.summary.by_resolver == {"openalex_oa": 1}
    assert (store.run_dir / "documents" / "A.pdf").read_bytes() == GOOD
    assert {a.validation for r in out.records for a in r.attempts} >= {"ok", "not_pdf", "no_text_layer"}


def test_limits_used_are_recorded_in_the_artifact(store):
    ctx = setup_run(store, three_papers()[:1], ["include"])
    make_stage(Server(ROUTES), max_papers=7, max_http_calls=33, pause=0.0,
               settings=DownloadSettings(max_bytes=10 * 1024 * 1024, min_title_coverage=0.5)).run(ctx)
    lim = outcome(store).limits
    assert (lim.max_papers, lim.max_http_calls, lim.max_mb, lim.min_title_coverage) == (7, 33, 10.0, 0.5)


def test_papers_follow_the_screening_order(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    assert [r.paper_id for r in outcome(store).records] == ["A", "B", "C"]


# ---------------------------------------------------------------- reprise
def test_second_run_does_no_download_for_finished_papers(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    again = Server({})
    result = make_stage(again).run(ctx)
    assert result.status == "waiting_human" and again.requests == []
    assert len(result.review_items) == 2


def test_deleted_pdf_is_downloaded_again(store):
    ctx = setup_run(store, three_papers()[:1], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    (store.run_dir / "documents" / "A.pdf").unlink()
    server = Server(ROUTES)
    assert make_stage(server).run(ctx).status == "done"
    assert len(server.requests) == 1 and (store.run_dir / "documents" / "A.pdf").exists()


def test_force_downloads_everything_again(store):
    ctx = setup_run(store, three_papers()[:1], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    server = Server(ROUTES)
    make_stage(server, force=True).run(ctx)
    assert len(server.requests) == 1


def test_retry_manual_retries_only_undecided_papers(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:C", "unavailable", "moi", "introuvable")
    server = Server({**ROUTES, "paywall.test": GOOD})  # B est devenu accessible
    result = make_stage(server, retry_manual=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"A": "downloaded", "B": "downloaded", "C": "unavailable"}
    assert not any("scan.test" in url for url in server.requests)  # C avait une décision : pas retenté


# ---------------------------------------------------------------- porte G3
def drop(store, pid, data):
    path = manual_pdf_path(store, pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_g3_provided_and_unavailable_complete_the_step(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    drop(store, "B", GOOD)
    ctx.queue.close_item("G3:paper:B", "provided", "moi", "PDF déposé")
    ctx.queue.close_item("G3:paper:C", "Unavailable", "moi", "introuvable")
    again = Server({})
    result = make_stage(again).run(ctx)
    assert result.status == "done" and again.requests == []
    assert statuses(store) == {"A": "downloaded", "B": "downloaded", "C": "unavailable"}
    out = outcome(store)
    b = next(r for r in out.records if r.paper_id == "B")
    assert b.attempts[-1].resolver == "manual" and b.attempts[-1].ok and b.pdf_path == "documents/B.pdf"
    assert out.summary.by_resolver == {"openalex_oa": 1, "manual": 1}
    assert out.summary.unavailable == 1


def test_manual_pdf_with_a_different_title_is_accepted_the_human_vouches_for_it(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    drop(store, "B", OTHER)
    ctx.queue.close_item("G3:paper:B", "provided", "moi", "c'est bien lui, titre différent")
    assert make_stage(Server({})).run(ctx).status == "done"


def test_provided_without_file_fails_then_can_be_fixed_by_dropping_the_file(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "provided", "moi", "je le dépose")
    result = make_stage(Server({})).run(ctx)
    assert result.status == "failed" and "aucun fichier" in result.message
    drop(store, "B", GOOD)  # le point est fermé pour toujours, mais le fichier peut être corrigé
    assert make_stage(Server({})).run(ctx).status == "done"


def test_provided_scanned_pdf_is_rejected(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    drop(store, "B", SCAN)
    ctx.queue.close_item("G3:paper:B", "provided", "moi", "x")
    result = make_stage(Server({})).run(ctx)
    assert result.status == "failed" and "couche texte" in result.message
    assert statuses(store) == {"B": "manual_required"}


def test_invalid_g3_decision_word_is_reported(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "approved", "moi", "oups")
    result = make_stage(Server({})).run(ctx)
    assert result.status == "failed" and "provided ou unavailable" in result.message


def test_closed_g3_decision_still_applies_after_force(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "unavailable", "moi", "introuvable")
    result = make_stage(Server(ROUTES), force=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"B": "unavailable"}


def test_manual_pdf_helpers(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    p = three_papers()[1]
    settings = DownloadSettings()
    assert "aucun fichier" in manual_pdf_problem(store, p, settings)
    drop(store, "B", HTML)
    assert check_manual_pdf(store, p, settings)[0] == "not_pdf"
    drop(store, "B", GOOD)
    assert check_manual_pdf(store, p, settings)[0] == "" and manual_pdf_problem(store, p, settings) == ""
    drop(store, "B", GOOD)
    assert check_manual_pdf(store, p, DownloadSettings(max_bytes=100))[0] == "too_large"


# ---------------------------------------------------------------- limites d'exécution
def test_max_papers_refuses_before_any_request(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    server = Server(ROUTES)
    result = make_stage(server, max_papers=2).run(ctx)
    assert result.status == "failed" and "--max-papers 2" in result.message
    assert server.requests == [] and not store.exists("documents")


def test_max_papers_counts_only_papers_still_to_download(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES), batch=2).run(ctx)  # 2 papiers faits, 1 reste
    assert make_stage(Server(ROUTES), max_papers=1).run(ctx).status == "waiting_human"


def test_batch_stops_cleanly_and_resumes(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    first = Server(ROUTES)
    result = make_stage(first, batch=2).run(ctx)
    assert result.status == "failed" and "lot de 2" in result.message and not store.exists("documents")
    assert set(store.read_artifact("documents_progress", DocumentsProgress).records) == {"A", "B"}
    second = Server(ROUTES)
    result = make_stage(second, batch=2).run(ctx)
    assert result.status == "waiting_human" and len(second.requests) == 1  # seul C est téléchargé
    assert set(statuses(store)) == {"A", "B", "C"}


def test_budget_exhaustion_keeps_progress_and_resumes(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    result = make_stage(Server(ROUTES), max_http_calls=1).run(ctx)
    assert result.status == "failed" and "budget" in result.message
    assert set(store.read_artifact("documents_progress", DocumentsProgress).records) == {"A"}
    assert make_stage(Server(ROUTES)).run(ctx).status == "waiting_human"
    assert set(statuses(store)) == {"A", "B", "C"}


def test_ctrl_c_saves_progress_and_reraises(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    with pytest.raises(KeyboardInterrupt):
        make_stage(Server({**ROUTES, "paywall.test": KeyboardInterrupt()})).run(ctx)
    assert set(store.read_artifact("documents_progress", DocumentsProgress).records) == {"A"}
    assert make_stage(Server(ROUTES)).run(ctx).status == "waiting_human"


def test_network_down_stops_without_recording_false_manual_papers(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    down = httpx.ConnectError("hors ligne")
    result = make_stage(Server({"good.test": down, "paywall.test": down, "scan.test": down}), max_unreachable=2).run(ctx)
    assert result.status == "failed" and "réseau inaccessible" in result.message
    assert store.read_artifact("documents_progress", DocumentsProgress).records == {}
    assert ctx.queue.open_items("G3") == [] and not store.exists("documents")
    assert make_stage(Server(ROUTES)).run(ctx).status == "waiting_human"  # le réseau est revenu


def test_unreachable_detection():
    from slrresearch.core.schemas.documents import DocumentRecord, DownloadAttempt as A
    net = DocumentRecord(paper_id="p", status="manual_required", attempts=[A(resolver="r", url="u", validation="http_error")])
    real = DocumentRecord(paper_id="p", status="manual_required", attempts=[A(resolver="r", url="u", http_status=403, validation="http_error")])
    empty = DocumentRecord(paper_id="p", status="manual_required")
    assert looks_unreachable(net) and not looks_unreachable(real) and not looks_unreachable(empty)


def test_progress_callback_reports_each_paper(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    seen = []
    make_stage(Server(ROUTES), progress=lambda i, n, pid, status: seen.append((i, n, pid, status))).run(ctx)
    assert seen == [(1, 3, "A", "downloaded (openalex_oa)"), (2, 3, "B", "manual_required"), (3, 3, "C", "manual_required")]


def test_trace_records_the_run(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    events = {e["event"] for e in ctx.trace.read()}
    assert {"started", "paper_done", "written"} <= events


# ---------------------------------------------------------------- schéma
def test_documents_outcome_rejects_inconsistent_counts():
    from pydantic import ValidationError
    from slrresearch.core.schemas.documents import DocumentRecord, DocumentsSummary, DownloadLimits
    limits = DownloadLimits(max_mb=1, max_http_calls=1, pause=0, min_title_coverage=0.5, min_chars_per_page=50)
    record = DocumentRecord(paper_id="a", status="manual_required")
    with pytest.raises(ValidationError):
        DocumentsOutcome(limits=limits, records=[record], summary=DocumentsSummary(total=1, downloaded=1, manual_required=0, unavailable=0))
    with pytest.raises(ValidationError):
        DocumentsOutcome(limits=limits, records=[record, record],
                         summary=DocumentsSummary(total=2, downloaded=0, manual_required=2, unavailable=0))


def test_force_with_retry_manual_still_applies_closed_g3_decisions(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "unavailable", "moi", "introuvable")
    result = make_stage(Server(ROUTES), force=True, retry_manual=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"B": "unavailable"}


def test_progress_is_saved_periodically_even_if_the_program_crashes(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    with pytest.raises(RuntimeError):
        make_stage(Server({**ROUTES, "scan.test": RuntimeError("plantage inattendu")}), save_every=1).run(ctx)
    assert set(store.read_artifact("documents_progress", DocumentsProgress).records) == {"A", "B"}


def test_retry_manual_that_fails_again_keeps_the_paper_waiting_for_g3(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    result = make_stage(Server(ROUTES), retry_manual=True).run(ctx)  # B est toujours derrière un paywall
    assert result.status == "waiting_human" and statuses(store) == {"B": "manual_required"}