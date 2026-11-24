"""Politique `on_missing` : un papier sans PDF en accès libre est déclaré indisponible sans intervention humaine."""
import httpx
import pytest

from slrresearch.core.schemas.documents import DocumentsOutcome
from slrresearch.download.stage import DownloadStage, manual_pdf_path
from tests.download_helpers import config
from tests.download_stage_helpers import setup_run
from tests.unit.test_download_stage import GOOD, ROUTES, Server, drop, make_stage, outcome, statuses, three_papers


def auto_stage(server, **kw):
    kw.setdefault("on_missing", "unavailable")
    return make_stage(server, **kw)


def test_missing_papers_become_unavailable_and_no_g3_point_is_opened(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    result = auto_stage(Server(ROUTES)).run(ctx)
    assert result.status == "done" and result.artifact == "documents"
    assert statuses(store) == {"A": "downloaded", "B": "unavailable", "C": "unavailable"}
    assert ctx.queue.open_items("G3") == [] and ctx.queue.list_items(gate="G3") == []
    out = outcome(store)
    assert out.limits.on_missing == "unavailable"
    assert (out.summary.downloaded, out.summary.unavailable, out.summary.manual_required) == (1, 2, 0)
    b = next(r for r in out.records if r.paper_id == "B")
    assert b.attempts[-1].resolver == "policy" and b.attempts[-1].validation == "unavailable:policy" and not b.attempts[-1].ok
    assert b.attempts[0].validation == "not_pdf"  # les vraies tentatives restent tracées
    assert "unavailable_by_policy" in {e["event"] for e in ctx.trace.read()}


def test_review_remains_the_default_of_the_stage(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    assert DownloadStage(config(email="")).on_missing == "review"
    assert make_stage(Server(ROUTES)).run(ctx).status == "waiting_human"


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        DownloadStage(config(), on_missing="ignore")


def test_open_g3_points_from_a_previous_review_run_are_closed_by_the_policy_without_network(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)  # mode review : 2 points G3 ouverts
    assert len(ctx.queue.open_items("G3")) == 2
    again = Server({})
    result = auto_stage(again).run(ctx)
    assert result.status == "done" and again.requests == []
    assert ctx.queue.open_items("G3") == []
    decisions = [d for d in ctx.queue.decisions.read() if d.gate == "G3"]
    assert {d.target_id for d in decisions} == {"B", "C"}
    assert all(d.decision == "unavailable" and d.decided_by.startswith("automatique") for d in decisions)
    assert all("on_missing=unavailable" in d.justification for d in decisions)
    assert statuses(store) == {"A": "downloaded", "B": "unavailable", "C": "unavailable"}


def test_an_earlier_human_decision_always_wins_over_the_policy(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    make_stage(Server(ROUTES)).run(ctx)
    drop(store, "B", GOOD)
    ctx.queue.close_item("G3:paper:B", "provided", "moi", "PDF déposé")
    result = auto_stage(Server({})).run(ctx)
    assert result.status == "done"
    assert statuses(store) == {"A": "downloaded", "B": "downloaded", "C": "unavailable"}
    humans = [d for d in ctx.queue.decisions.read() if d.decided_by == "moi"]
    assert [d.target_id for d in humans] == ["B"]


def test_a_human_decision_with_an_invalid_word_is_still_reported_not_overridden(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "approved", "moi", "oups")
    result = auto_stage(Server({})).run(ctx)
    assert result.status == "failed" and "provided ou unavailable" in result.message


def test_retry_manual_also_retries_papers_declared_unavailable_by_the_policy(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    auto_stage(Server(ROUTES)).run(ctx)
    server = Server({**ROUTES, "paywall.test": GOOD})  # B est devenu accessible
    result = auto_stage(server, retry_manual=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"A": "downloaded", "B": "downloaded", "C": "unavailable"}
    assert sum(1 for url in server.requests if "paywall.test" in url) == 1
    assert sum(1 for url in server.requests if "scan.test" in url) == 1  # C est retenté aussi, et échoue de nouveau


def test_without_retry_manual_policy_decisions_are_final(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    auto_stage(Server(ROUTES)).run(ctx)
    again = Server({**ROUTES, "paywall.test": GOOD})
    auto_stage(again).run(ctx)
    assert again.requests == [] and statuses(store)["B"] == "unavailable"


def test_a_human_unavailable_decision_is_never_retried(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "unavailable", "moi", "introuvable")
    server = Server({**ROUTES, "paywall.test": GOOD})
    result = auto_stage(server, retry_manual=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"B": "unavailable"} and server.requests == []


def test_network_failure_is_not_turned_into_an_unavailable_paper(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    down = httpx.ConnectError("hors ligne")
    result = auto_stage(Server({"good.test": down, "paywall.test": down, "scan.test": down}), max_unreachable=2).run(ctx)
    assert result.status == "failed" and "réseau inaccessible" in result.message
    assert not store.exists("documents")
    assert auto_stage(Server(ROUTES)).run(ctx).status == "done"


def test_policy_applies_with_batches_and_resumes(store):
    ctx = setup_run(store, three_papers(), ["include"] * 3)
    assert auto_stage(Server(ROUTES), batch=2).run(ctx).status == "failed"
    assert not store.exists("documents")
    assert auto_stage(Server(ROUTES), batch=2).run(ctx).status == "done"
    assert statuses(store) == {"A": "downloaded", "B": "unavailable", "C": "unavailable"}


def test_old_artifacts_without_the_policy_field_still_load():
    from slrresearch.core.schemas.documents import DocumentRecord, DocumentsSummary, DownloadLimits
    limits = DownloadLimits.model_validate({"max_mb": 50, "max_http_calls": 400, "pause": 1, "min_title_coverage": 0.7,
                                            "min_chars_per_page": 50})
    assert limits.on_missing == "review"


def test_a_human_unavailable_already_saved_is_not_retried_on_a_later_run(store):
    ctx = setup_run(store, three_papers()[1:2], ["include"])
    make_stage(Server(ROUTES)).run(ctx)
    ctx.queue.close_item("G3:paper:B", "unavailable", "moi", "introuvable")
    auto_stage(Server({})).run(ctx)  # la décision humaine est enregistrée dans l'état du papier
    server = Server({**ROUTES, "paywall.test": GOOD})
    result = auto_stage(server, retry_manual=True).run(ctx)
    assert result.status == "done" and statuses(store) == {"B": "unavailable"} and server.requests == []