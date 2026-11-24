from slrresearch.core.schemas.facet_evidence import FacetEvidenceRecord
from slrresearch.core.schemas.facet_plan import CanonicalFacet
from slrresearch.core.schemas.fidelity import FidelityReport
from slrresearch.core.schemas.study_card import CardField
from slrresearch.evidence.facet_evidence import (
    _keep_prior_after_repair_error, _record_from_new, _record_from_reused,
    _status_from_card_field, load_fidelity_prompt,
)
from slrresearch.extraction.study_card import load_prompt


def facet(category="parameters", source_ids=None, rq_ids=None, reuse_field=""):
    return CanonicalFacet(id=category, label=category.title(), description="A normalized evidence facet.",
        query_en="english query terms", blocking=False, reuse_field=reuse_field,
        source_facet_ids=source_ids or ["F5"], rq_ids=rq_ids or ["RQ1"])


def test_source_card_statuses_map_to_explicit_facet_states():
    def field(status):
        note = "source note" if status not in ("verified", "not_reported") else ""
        detail = {"value": "value", "chunk_id": "P1_p1_c1", "page": 1, "quote": "supporting quote"} if status == "verified" else {}
        return CardField(field_id="parameters", label="Parameters", status=status, note=note, **detail)
    assert _status_from_card_field(field("not_reported")) == "not_reported"
    assert _status_from_card_field(field("rejected")) == "rejected"
    assert _status_from_card_field(field("llm_error")) == "llm_error"
    assert _status_from_card_field(field("no_context")) == "missing"


def test_new_not_reported_field_does_not_call_or_require_judge():
    field = CardField(field_id="parameters", label="Parameters", status="not_reported")
    record, item = _record_from_new("P1", facet(), field, "judge", None, "", [], "fp", None)
    assert isinstance(record, FacetEvidenceRecord)
    assert record.status == "not_reported"
    assert record.fidelity is None
    assert item is None


def test_reused_approved_card_field_links_all_source_facets():
    f = facet("models", ["F3", "F9", "F15"], ["RQ1", "RQ2", "RQ3"], "models")
    source = CardField(field_id="models", label="Modèles", status="verified", value="GPT-4",
        chunk_id="P1_p1_c1", page=1, quote="We evaluated GPT-4.")
    report = FidelityReport(paper_id="P1", field_id="models", value="GPT-4", chunk_id="P1_p1_c1", page=1,
        quote="We evaluated GPT-4.", quote_in_chunk=True, provenance_ok=True, numbers_ok=True,
        entailment="yes", own_study=True, judge_model="judge", status="verified")
    record, items = _record_from_reused("P1", f, source, report, None)
    assert record.status == "found"
    assert record.reused
    assert record.source_facet_ids == ["F3", "F9", "F15"]
    assert record.value == "GPT-4"
    assert items == []


def test_facet_stage_loads_distinct_extraction_and_judge_prompts():
    assert "INFORMATION DEMANDÉE" in load_prompt()
    judge_prompt = load_fidelity_prompt()
    assert "TÂCHE 1" in judge_prompt
    assert "unsupported_claims" in judge_prompt


def test_transient_repair_error_preserves_the_previous_reviewed_status():
    prior = FacetEvidenceRecord(paper_id="P1", facet_id="design", source_facet_ids=["F1"], rq_ids=["RQ1"],
        status="rejected", value="old value", quote="old quote", note="Previously rejected quote.")
    failed = prior.model_copy(update={"status": "llm_error", "value": "", "quote": "",
        "note": "Connection error", "repair_attempted": True})
    kept = _keep_prior_after_repair_error(prior, failed)
    assert kept.status == "rejected"
    assert kept.value == "old value"
    assert kept.note == "Previously rejected quote."
    assert not kept.repair_attempted
