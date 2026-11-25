import json

import pytest

from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.extraction import (
    Chunk,
    ExtractedPaperEntry,
    ExtractionLimits,
    ExtractionOutcome,
    ExtractionSummary,
    ExtractionParams,
    PaperText,
)
from slrresearch.core.schemas.facet_evidence import (
    FacetEvidenceOutcome,
    FacetEvidencePaper,
    FacetEvidenceRecord,
    FacetEvidenceSummary,
)
from slrresearch.core.schemas.facet_matrix import (
    FacetMatrixCell,
    FacetMatrixOutcome,
    FacetMatrixPaper,
    FacetMatrixSummary,
)
from slrresearch.core.schemas.facet_plan import (
    CanonicalFacet,
    FacetPlanOutcome,
    FacetPlanSource,
)
from slrresearch.core.schemas.quality import QualityOutcome
from slrresearch.core.store import RunStore
from slrresearch.evidence.fidelity import ROLE as JUDGE_ROLE
from slrresearch.hitl.decisions import DecisionLog
from slrresearch.hitl.queue import ReviewQueue
from slrresearch.quality.checklist import CHECKLIST, parse_response
from slrresearch.quality.stage import QualityStage
from tests.fakes import FakeLLM


PAPER_ID = "P1"
PLAN_HASH = "a" * 64
QUOTE = "The study evaluated the screening workflow on a labeled dataset."
CATEGORIES = (
    "design", "metrics", "models", "data", "parameters", "results",
    "risks_limits", "methodology",
)


def response_payload(overrides=None):
    overrides = overrides or {}
    assessments = []
    for criterion in CHECKLIST:
        assessment = {
            "criterion_id": criterion.id,
            "rating": "yes",
            "confidence": 0.95,
            "rationale": "The excerpt supports this criterion.",
            "quote": QUOTE,
            "chunk_id": f"{PAPER_ID}_p1_c1",
        }
        assessment.update(overrides.get(criterion.id, {}))
        assessments.append(assessment)
    return {"assessments": assessments}


def make_context(store, llm):
    settings = Settings.from_env({
        "LLM_PROVIDER": "groq",
        "GROQ_MODEL": "extractor-model",
        "GROQ_MODEL_JUDGE": "judge-model",
    })
    return build_context(store, settings, make_model=lambda _settings, _role: llm)


def prepare_run(store, evidence_status="found", blocking_missing=0):
    sources = []
    facets = []
    evidence_rows = []
    cells = []
    for index, category in enumerate(CATEGORIES, start=1):
        facet_id = f"F{index}"
        sources.append(FacetPlanSource(
            facet_id=facet_id,
            rq_id="RQ1",
            original_label=category,
            original_description=f"Description for {category}",
            category=category,
        ))
        facets.append(CanonicalFacet(
            id=category,
            label=f"Facet {category}",
            description=f"Description for {category}",
            query_en=f"Evidence about {category}",
            blocking=True,
            source_facet_ids=[facet_id],
            rq_ids=["RQ1"],
        ))
        status = evidence_status if category == "data" else "found"
        evidence_rows.append(FacetEvidenceRecord(
            paper_id=PAPER_ID,
            facet_id=category,
            source_facet_ids=[facet_id],
            rq_ids=["RQ1"],
            status=status,
            value="A labeled dataset" if status == "found" else "",
            quote=QUOTE if status == "found" else "",
            chunk_id=f"{PAPER_ID}_p1_c1" if status == "found" else "",
            page=1 if status == "found" else None,
        ))
        cell_status = "missing" if blocking_missing and category == "data" else (
            status if status in ("found", "not_reported", "missing") else "missing"
        )
        cells.append(FacetMatrixCell(
            paper_id=PAPER_ID,
            facet_id=facet_id,
            rq_id="RQ1",
            category=category,
            blocking=True,
            status=cell_status,
            evidence_status=status,
        ))

    plan = FacetPlanOutcome(
        version="test",
        protocol_sha256=PLAN_HASH,
        gate_id="G1:facet_plan:test",
        source_facets=sources,
        facets=facets,
    )
    evidence = FacetEvidenceOutcome(
        version="test",
        plan_hash=PLAN_HASH,
        judge_model="judge-model",
        extraction_model="extractor-model",
        papers=[FacetEvidencePaper(paper_id=PAPER_ID, evidence=evidence_rows)],
        summary=FacetEvidenceSummary(
            papers=1,
            found=sum(row.status == "found" for row in evidence_rows),
            not_reported=sum(row.status == "not_reported" for row in evidence_rows),
            missing=sum(row.status == "missing" for row in evidence_rows),
            human_review=sum(row.status == "human_review" for row in evidence_rows),
            rejected=sum(row.status == "rejected" for row in evidence_rows),
            llm_error=sum(row.status == "llm_error" for row in evidence_rows),
            reused=0,
            llm_calls=0,
            tokens=0,
        ),
    )
    matrix = FacetMatrixOutcome(
        version="test",
        plan_hash=PLAN_HASH,
        papers=[FacetMatrixPaper(paper_id=PAPER_ID, cells=cells)],
        summary=FacetMatrixSummary(
            papers=1,
            cells=len(cells),
            found=sum(cell.status == "found" for cell in cells),
            not_reported=sum(cell.status == "not_reported" for cell in cells),
            missing=sum(cell.status == "missing" for cell in cells),
            contradictory=0,
            blocking_missing=sum(cell.blocking and cell.status == "missing" for cell in cells),
        ),
    )

    text = "The study evaluated the screening workflow on a labeled dataset."
    chunk = Chunk(paper_id=PAPER_ID, chunk_id=f"{PAPER_ID}_p1_c1", page=1, text=text)
    extraction_dir = store.run_dir / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    paper_text = PaperText(
        paper_id=PAPER_ID,
        source_sha256="b" * 64,
        params=ExtractionParams(
            chunk_size=500,
            chunk_overlap=0,
            min_chars_per_page=0,
            cleaning="test",
        ),
        total_pages=1,
        pages_read=1,
        page_chars=[len(text)],
        chunks=[chunk],
    )
    (extraction_dir / f"{PAPER_ID}.json").write_text(
        paper_text.model_dump_json(), encoding="utf-8"
    )
    extraction = ExtractionOutcome(
        limits=ExtractionLimits(
            chunk_size=500,
            chunk_overlap=0,
            min_chars_per_page=0,
            cleaning="test",
        ),
        papers=[ExtractedPaperEntry(
            paper_id=PAPER_ID,
            status="extracted",
            file=f"extraction/{PAPER_ID}.json",
            source_sha256="b" * 64,
            total_pages=1,
            pages_read=1,
            chars=len(text),
            chunks=1,
        )],
        summary=ExtractionSummary(
            total=1, extracted=1, failed=0, chunks=1, skipped_unavailable=0
        ),
    )
    store.write_artifact("facet_plan", FacetPlanOutcome, plan)
    store.write_artifact("facet_evidence", FacetEvidenceOutcome, evidence)
    store.write_artifact("facet_matrix", FacetMatrixOutcome, matrix)
    store.write_artifact("extraction", ExtractionOutcome, extraction)


def test_parse_response_requires_one_valid_assessment_per_criterion():
    parsed = parse_response(json.dumps(response_payload()))
    assert {item.criterion_id for item in parsed.assessments} == set(
        criterion.id for criterion in CHECKLIST
    )

    duplicate = response_payload()
    duplicate["assessments"][-1]["criterion_id"] = "Q1"
    with pytest.raises(ValueError, match="exactement une évaluation"):
        parse_response(json.dumps(duplicate))


def test_parse_response_rejects_not_reported_with_a_quote():
    payload = response_payload({"Q2": {"rating": "not_reported"}})
    with pytest.raises(ValueError, match="not_reported ne doit pas inventer"):
        parse_response(json.dumps(payload))


def test_context_does_not_pass_unaccepted_evidence_as_factual_excerpt(tmp_path):
    from slrresearch.quality.stage import _evidence_context

    store = RunStore(tmp_path, "quality_context_test")
    prepare_run(store, evidence_status="rejected")
    evidence = store.read_artifact("facet_evidence", FacetEvidenceOutcome)
    context = _evidence_context(evidence.papers[0])
    data = next(item for item in context if item["criterion_id"] == "Q2")
    assert data["evidence"] == [{
        "category": "data",
        "status": "rejected",
        "note": "No accepted excerpt supplied for this category.",
    }]
    assert QUOTE not in json.dumps(data)


def test_quality_stage_accepts_cited_yes_ratings_without_human_gate(tmp_path):
    store = RunStore(tmp_path, "run1")
    prepare_run(store)
    llm = FakeLLM([json.dumps(response_payload())])

    result = QualityStage(pause=0).run(make_context(store, llm))

    outcome = store.read_artifact("quality", QualityOutcome)
    assert result.status == "done"
    assert outcome.summary.complete == 1
    assert outcome.summary.accepted == len(CHECKLIST)
    assert outcome.summary.llm_errors == 0
    assert llm.calls == 1


@pytest.mark.parametrize(
    ("decision", "expected_paper_status", "expected_item_status"),
    [("approved", "complete", "accepted"), ("rejected", "rejected", "rejected")],
)
def test_g4_decision_is_applied_from_cache_without_another_llm_call(
    tmp_path, decision, expected_paper_status, expected_item_status
):
    store = RunStore(tmp_path, "run1")
    prepare_run(store)
    payload = response_payload({
        "Q2": {
            "rating": "partial",
            "confidence": 0.8,
            "rationale": "Only part of the dataset details are shown.",
        }
    })
    llm = FakeLLM([json.dumps(payload)])
    ctx = make_context(store, llm)

    first = QualityStage(pause=0).run(ctx)
    first_outcome = store.read_artifact("quality", QualityOutcome)
    gate_id = first_outcome.papers[0].review_item_id
    assert first.status == "waiting_human"
    assert gate_id.startswith("G4:quality:")

    ctx.queue.close_item(gate_id, decision, "reviewer", "Reviewed the flagged rating.")
    second = QualityStage(pause=0).run(ctx)
    outcome = store.read_artifact("quality", QualityOutcome)

    assert second.status == "done"
    assert outcome.papers[0].status == expected_paper_status
    reviewed = next(item for item in outcome.papers[0].assessments if item.criterion_id == "Q2")
    assert reviewed.status == expected_item_status
    assert reviewed.reviewer == "reviewer"
    assert llm.calls == 1


def test_quality_stage_rejects_quote_not_present_in_source_chunk(tmp_path):
    store = RunStore(tmp_path, "run1")
    prepare_run(store)
    payload = response_payload({"Q1": {"quote": "A fabricated citation."}})
    llm = FakeLLM([json.dumps(payload)])

    result = QualityStage(pause=0).run(make_context(store, llm))
    outcome = store.read_artifact("quality", QualityOutcome)

    assert result.status == "failed"
    assert outcome.summary.incomplete == 1
    assert outcome.summary.llm_errors == 1
    bad = next(item for item in outcome.papers[0].assessments if item.criterion_id == "Q1")
    assert bad.status == "llm_error"
    assert not bad.quote_in_chunk
    assert llm.calls == 1


def test_quality_stage_refuses_matrix_with_blocking_missing_facet(tmp_path):
    store = RunStore(tmp_path, "run1")
    prepare_run(store, blocking_missing=1)
    llm = FakeLLM([])

    result = QualityStage(pause=0).run(make_context(store, llm))

    assert result.status == "failed"
    assert "facettes manquantes" in result.message
    assert llm.calls == 0


def test_quality_cli_runs_stage_and_prints_summary(tmp_path, capsys, monkeypatch):
    import cli

    store = RunStore(tmp_path, "run1")
    prepare_run(store)
    monkeypatch.setattr(cli, "load_settings", lambda: Settings.from_env({
        "LLM_PROVIDER": "groq",
        "GROQ_MODEL": "extractor-model",
        "GROQ_MODEL_JUDGE": "judge-model",
    }))
    llm = FakeLLM([json.dumps(response_payload())])

    code = cli.main(
        ["--root", str(tmp_path), "quality", "--run-id", "run1", "--pause", "0"],
        make_model=lambda _settings, _role: llm,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "OK : étape terminée, artefact « quality »." in output
    assert "Articles : 1 ; terminés : 1 ; revue G4 : 0" in output
    assert llm.calls == 1
