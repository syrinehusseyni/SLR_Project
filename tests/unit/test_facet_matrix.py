import pytest

from slrresearch.core.schemas.facet_evidence import (
    FacetEvidenceOutcome, FacetEvidencePaper, FacetEvidenceRecord, FacetEvidenceSummary,
)
from slrresearch.core.schemas.facet_plan import (
    CanonicalFacet, FacetPlanOutcome, FacetPlanSource,
)
from slrresearch.evidence.facet_matrix import build_facet_matrix


CATEGORIES = ["design", "metrics", "models", "data", "parameters", "results", "risks_limits", "methodology"]


def inputs(statuses=None):
    statuses = statuses or {}
    sources, facets, records = [], [], []
    for i, category in enumerate(CATEGORIES, 1):
        facet_id = f"F{i}"
        sources.append(FacetPlanSource(facet_id=facet_id, rq_id="RQ1", original_label=category,
            original_description=f"Description for {category}", category=category))
        facets.append(CanonicalFacet(id=category, label=f"Facet {category}", description=f"Description for {category}",
            query_en=f"Search query for {category}", blocking=category != "data", source_facet_ids=[facet_id],
            rq_ids=["RQ1"]))
        status = statuses.get(category, "missing")
        records.append(FacetEvidenceRecord(paper_id="P1", facet_id=category, source_facet_ids=[facet_id],
            rq_ids=["RQ1"], status=status,
            value="value" if status == "found" else "", quote="support" if status == "found" else "",
            chunk_id="P1_p1_c1" if status == "found" else "", page=1 if status == "found" else None))
    plan = FacetPlanOutcome(version="test", protocol_sha256="a" * 64, gate_id="G1:facet_plan:test",
        source_facets=sources, facets=facets)
    evidence = FacetEvidenceOutcome(version="test", plan_hash="a" * 64, judge_model="judge", extraction_model="extractor",
        papers=[FacetEvidencePaper(paper_id="P1", evidence=records)],
        summary=FacetEvidenceSummary(papers=1,
            found=sum(r.status == "found" for r in records),
            not_reported=sum(r.status == "not_reported" for r in records),
            missing=sum(r.status == "missing" for r in records),
            human_review=sum(r.status == "human_review" for r in records),
            rejected=sum(r.status == "rejected" for r in records),
            llm_error=sum(r.status == "llm_error" for r in records), reused=0, llm_calls=0, tokens=0))
    return plan, evidence


def test_matrix_expands_canonical_results_to_original_protocol_facets():
    plan, evidence = inputs({"design": "found", "metrics": "not_reported"})
    matrix = build_facet_matrix(plan, evidence)
    cells = {cell.facet_id: cell for cell in matrix.papers[0].cells}
    assert cells["F1"].status == "found"
    assert cells["F1"].quote == "support"
    assert cells["F2"].status == "not_reported"
    assert matrix.summary.cells == 8
    assert matrix.summary.found == 1
    assert matrix.summary.not_reported == 1
    assert matrix.summary.blocking_missing == 5


@pytest.mark.parametrize("source_status", ["rejected", "human_review", "llm_error"])
def test_unaccepted_or_unresolved_evidence_is_not_counted_as_coverage(source_status):
    plan, evidence = inputs({"design": source_status})
    matrix = build_facet_matrix(plan, evidence)
    cell = matrix.papers[0].cells[0]
    assert cell.status == "missing"
    assert cell.evidence_status == source_status
    assert cell.blocking


def test_matrix_refuses_evidence_generated_for_a_different_plan():
    plan, evidence = inputs()
    evidence.plan_hash = "b" * 64
    with pytest.raises(ValueError, match="ne correspond pas"):
        build_facet_matrix(plan, evidence)
