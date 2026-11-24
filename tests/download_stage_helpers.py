"""Aides pour tester l'étape 4 : un run avec les artefacts `search` et `screening` déjà prêts."""
from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.screening import (
    AgreementStats, RuleVerdict, ScreeningOutcome, ScreeningRecord, ScreeningSummary,
)
from slrresearch.core.schemas.search import QueryRun, SearchOutcome, SearchSummary
from tests.fakes import FakeLLM
from tests.pdf_helpers import TITLE


def paper(pid, title=TITLE, **kw):
    return Paper(id=pid, title=title, source="openalex", **kw)


def _record(p, kind):
    rule = RuleVerdict(paper_id=p.id, decision="include" if kind != "exclude" else "exclude", reasons=["test"], decisive=True)
    if kind == "include":
        return ScreeningRecord(paper_id=p.id, title=p.title, rule=rule, llm_status="not_run", final="include", origin="rule")
    if kind == "exclude":
        return ScreeningRecord(paper_id=p.id, title=p.title, rule=rule, llm_status="not_run", final="exclude", origin="rule")
    return ScreeningRecord(paper_id=p.id, title=p.title, rule=rule, llm_status="not_run", final="uncertain",
                           origin="pending", escalation="en attente")


def setup_run(store, papers, kinds):
    """papers : liste de Paper ; kinds : 'include' | 'exclude' | 'pending' pour chacun (dans l'ordre)."""
    n = len(papers)
    store.write_artifact("search", SearchOutcome, SearchOutcome(
        runs=[QueryRun(source="openalex", query_param="q", query="x", status="ok", retrieved=n)],
        papers=papers, found_in={p.id: ["openalex"] for p in papers},
        summary=SearchSummary(per_source={"openalex": n}, retrieved=n, duplicates_removed=0, unique=n, without_abstract=n),
    ).model_dump())
    records = [_record(p, k) for p, k in zip(papers, kinds)]
    by_origin = {}
    for r in records:
        by_origin[r.origin] = by_origin.get(r.origin, 0) + 1
    store.write_artifact("screening", ScreeningOutcome, ScreeningOutcome(
        topic="t", prompt="screening.txt", fingerprint="f", min_confidence=0.6, records=records,
        summary=ScreeningSummary(total=n, included=kinds.count("include"), excluded=kinds.count("exclude"),
                                 pending=kinds.count("pending"), by_origin=by_origin, without_abstract=n, llm_failures=0,
                                 agreement=AgreementStats(n=0, agreed=0)),
    ).model_dump())
    return build_context(store, Settings.from_env({}), make_model=lambda s, r: FakeLLM([]))