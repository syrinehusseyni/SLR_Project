import json

from slrresearch.core.llm import LLMClient
from slrresearch.core.schemas.extraction import Chunk
from slrresearch.core.schemas.study_card import CardField
from slrresearch.core.schemas.fidelity import FidelityReport
from slrresearch.core.schemas.protocol import ResearchQuestion
from slrresearch.core.schemas.protocol_plan import (
    Criteria, Criterion, FacetSpec, Protocol, SearchConcept, SearchPlan, SourceQuery,
)
from slrresearch.evidence.fidelity import _apply_human_decision, _report
from slrresearch.evidence.facet_plan import build_facet_plan
from slrresearch.hitl.decisions import DecisionLog
from slrresearch.hitl.queue import ReviewQueue
from tests.fakes import FakeLLM


def make_field(value="0.82 accuracy", quote="The model achieved 0.82 accuracy."):
    return CardField(field_id="findings", label="Résultats", status="verified", value=value,
                     chunk_id="P1_p1_c1", page=1, quote=quote)


def make_chunk(text="The model achieved 0.82 accuracy."):
    return Chunk(paper_id="P1", chunk_id="P1_p1_c1", page=1, text=text)


def test_fidelity_judge_accepts_supported_own_study():
    llm = FakeLLM([json.dumps({"entailment": "yes", "own_study": True, "unsupported_claims": [], "reason": "Correspond."})])
    client = LLMClient(llm, "judge-model", "judge")
    report = _report("P1", make_field(), make_chunk(), "judge-model", client, "{{context}}", "fp")
    assert report.status == "verified"
    assert report.quote_in_chunk and report.provenance_ok and report.numbers_ok
    assert len(llm.prompts) == 1


def test_fidelity_partial_requires_human_review():
    llm = FakeLLM([json.dumps({"entailment": "partial", "own_study": True, "unsupported_claims": ["portée"], "reason": "Portée imprécise."})])
    report = _report("P1", make_field(), make_chunk(), "judge-model", LLMClient(llm, "judge-model", "judge"), "x", "fp")
    assert report.status == "human_review"


def test_fidelity_rejects_quote_missing_from_chunk_without_calling_judge():
    llm = FakeLLM([])
    report = _report("P1", make_field(), make_chunk("Different passage."), "judge-model",
                     LLMClient(llm, "judge-model", "judge"), "x", "fp")
    assert report.status == "rejected"
    assert not report.quote_in_chunk
    assert llm.calls == 0


def test_fidelity_rejects_unsupported_number_without_calling_judge():
    llm = FakeLLM([])
    field = make_field("0.95 accuracy")
    report = _report("P1", field, make_chunk(), "judge-model", LLMClient(llm, "judge-model", "judge"), "x", "fp")
    assert report.status == "rejected"
    assert not report.numbers_ok
    assert llm.calls == 0


def test_fidelity_rejects_judgment_about_other_study():
    llm = FakeLLM([json.dumps({"entailment": "yes", "own_study": False, "unsupported_claims": [], "reason": "Travail cité."})])
    report = _report("P1", make_field(), make_chunk(), "judge-model", LLMClient(llm, "judge-model", "judge"), "x", "fp")
    assert report.status == "rejected"


def test_quote_only_guard_routes_unsupported_cross_validation_to_human():
    value = ("Étude comparative expérimentale avec entraînement et validation croisée.")
    quote = "We evaluate inclusion and exclusion performance through accuracy, precision and recall."
    field = CardField(field_id="design", label="Type d'étude", status="verified", value=value,
                      chunk_id="P1_p1_c1", page=1, quote=quote)
    llm = FakeLLM([json.dumps({"entailment": "yes", "own_study": True, "unsupported_claims": [], "reason": "Tout est soutenu."})])
    report = _report("P1", field, make_chunk(quote), "judge-model", LLMClient(llm, "judge-model", "judge"), "x", "fp")
    assert report.status == "human_review"
    assert report.entailment == "partial"
    assert "validation croisée" in report.reason


def test_quote_only_guard_accepts_comparably_as_comparative_support():
    from slrresearch.evidence.fidelity import _unsupported_known_claims
    assert _unsupported_known_claims("Étude comparative", "The package performs comparably to zero-shot.") == []


def test_quote_only_guard_catches_comparison_claim_missing_from_quote():
    from slrresearch.evidence.fidelity import _unsupported_known_claims
    assert _unsupported_known_claims("expérience comparative", "We ran the experiment in triplicate.") == ["expérience comparative"]


def make_review_report():
    return FidelityReport(paper_id="P1", field_id="design", value="value", chunk_id="P1_p1_c1", page=1,
        quote="quote", quote_in_chunk=True, provenance_ok=True, numbers_ok=True, entailment="partial",
        own_study=True, reason="partial", judge_model="judge-model", input_fingerprint="fp", status="human_review")


def test_closed_g5_approval_updates_report_to_verified_and_is_recorded(tmp_path):
    queue = ReviewQueue(tmp_path / "hitl.jsonl", DecisionLog(tmp_path / "decisions.jsonl"))
    queue.open_item("G5", "evidence", "P1:design", "partial")
    queue.close_item("G5:evidence:P1:design", "approved", "reviewer", "citation supports value")
    report = _apply_human_decision(make_review_report(), queue)
    assert report.status == "verified"
    assert report.human_decision == "approved"
    assert report.reviewer == "reviewer"


def test_closed_g5_rejection_updates_report_to_rejected(tmp_path):
    queue = ReviewQueue(tmp_path / "hitl.jsonl", DecisionLog(tmp_path / "decisions.jsonl"))
    queue.open_item("G5", "evidence", "P1:design", "partial")
    queue.close_item("G5:evidence:P1:design", "rejected", "reviewer", "quote is insufficient")
    assert _apply_human_decision(make_review_report(), queue).status == "rejected"


def test_facet_plan_normalizes_18_source_facets_to_8_categories():
    questions = [ResearchQuestion(identifier=f"RQ{i}", question=f"Research question number {i} asks something useful?") for i in range(1, 4)]
    facets = []
    for i in range(1, 19):
        rq = 1 if i <= 6 else 2 if i <= 12 else 3
        facets.append(FacetSpec(id=f"F{i}", rq_id=f"RQ{rq}", label="Facet label", description="Long enough facet description", blocking=True))
    protocol = Protocol(
        topic="A topic",
        questions=questions,
        concepts=[SearchConcept(id="C1", label="First concept", terms=["large language model", "LLM"]),
                  SearchConcept(id="C2", label="Second concept", terms=["screening", "systematic review"])],
        criteria=Criteria(inclusion=[Criterion(id="I1", kind="include", text="Include qualifying studies.")],
                          exclusion=[Criterion(id="E1", kind="exclude", text="Exclude unrelated studies.")]),
        facets=facets,
        search_plan=SearchPlan(queries=[SourceQuery(source="openalex", query_param="search", query="LLM screening")]),
    )
    plan = build_facet_plan(protocol)
    assert len(plan.source_facets) == 18
    assert len(plan.facets) == 8
    assert {facet.id for facet in plan.facets} == {
        "design", "metrics", "models", "data", "parameters", "results", "risks_limits", "methodology"
    }
    assert {source for facet in plan.facets for source in facet.source_facet_ids} == {f"F{i}" for i in range(1, 19)}
    assert all(facet.query_en.isascii() for facet in plan.facets)
