import json

import pytest

from slrresearch.core.config import Settings
from slrresearch.core.contract import StageResult, build_context
from slrresearch.core.errors import ArtifactInvalid
from slrresearch.core.schemas.protocol import QuestionsOutcome, ResearchQuestion
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.protocol.questions_stage import QuestionsStage
from slrresearch.protocol.stage import APPROVED, ProtocolStage
from tests.fakes import FakeLLM

Q_TEXT = (
    "RQ1. Quelle précision obtiennent les LLM au screening ?\n"
    "RQ2. Quelles limites sont rapportées dans les études ?\n"
    "RQ3. Quels modèles sont comparés dans la littérature ?\n"
)
CONCEPTS = "task | title screening; abstract screening; study selection\nmethod | large language model; LLM; GPT-4\n"
CRITERIA = (
    "INCLUDE | L'étude évalue un LLM pour le screening\nINCLUDE | L'étude rapporte au moins une métrique\n"
    "EXCLUDE | L'article n'est pas évalué empiriquement\nEXCLUDE | L'article est une simple opinion\n"
)
FACETS = (
    "RQ1 | Métrique | Mesure de performance rapportée | oui\nRQ2 | Limites | Limites citées par les auteurs | oui\n"
    "RQ3 | Modèles | Modèles comparés dans l'étude | non\n"
)


def make_ctx(store, responses):
    fake = FakeLLM(responses)
    settings = Settings.from_env({"LLM_PROVIDER": "ollama"})
    return build_context(store, settings, make_model=lambda s, role: fake), fake


def write_questions(store):
    outcome = QuestionsOutcome(
        topic="LLM et screening", status="accepted", text=Q_TEXT,
        questions=[ResearchQuestion(identifier=f"RQ{i}", question=q) for i, q in
                   enumerate(["Quelle précision obtiennent les LLM au screening ?",
                              "Quelles limites sont rapportées dans les études ?",
                              "Quels modèles sont comparés dans la littérature ?"], start=1)],
    )
    store.write_artifact("questions", QuestionsOutcome, outcome.model_dump())


# ---------- QuestionsStage ----------
def test_questions_stage_writes_artifact(store):
    ctx, fake = make_ctx(store, [Q_TEXT])
    result = QuestionsStage("LLM et screening").run(ctx)
    assert result.status == "done" and result.artifact == "questions"
    assert store.artifact_path("questions").exists()  # c'est CE fichier qui manquait dans runs/
    assert store.read_artifact("questions", QuestionsOutcome).status == "accepted"


def test_questions_stage_reuses_accepted_artifact(store):
    ctx, fake = make_ctx(store, [Q_TEXT])
    QuestionsStage("LLM et screening").run(ctx)
    QuestionsStage("LLM et screening").run(ctx)
    assert fake.calls == 1  # pas de second appel LLM


def test_questions_stage_failed_when_invalid(store):
    ctx, fake = make_ctx(store, ["mauvais", "encore mauvais"])
    result = QuestionsStage("x").run(ctx)
    assert result.status == "failed" and "invalides" in result.message
    # relancer régénère (un artefact human_required n'est pas réutilisé)
    ctx2, fake2 = make_ctx(store, [Q_TEXT])
    assert QuestionsStage("x").run(ctx2).status == "done"


# ---------- ProtocolStage : contrat avec l'artefact questions ----------
def test_protocol_fails_without_questions(store):
    ctx, fake = make_ctx(store, [])
    result = ProtocolStage().run(ctx)
    assert result.status == "failed" and "questions" in result.message and fake.calls == 0


def test_protocol_fails_if_questions_human_required(store):
    store.write_artifact("questions", QuestionsOutcome, QuestionsOutcome(topic="x", status="human_required", text="n'importe quoi", error="e").model_dump())
    ctx, fake = make_ctx(store, [])
    assert ProtocolStage().run(ctx).status == "failed" and fake.calls == 0


def test_questions_artifact_feeds_protocol(store):
    ctx, fake = make_ctx(store, [Q_TEXT, CONCEPTS, CRITERIA, FACETS])
    assert QuestionsStage("LLM et screening").run(ctx).status == "done"
    result = ProtocolStage(year_from=2018).run(ctx)
    assert result.status == "waiting_human" and fake.calls == 4
    protocol = store.read_artifact("protocol", Protocol)
    assert [q.identifier for q in protocol.questions] == ["RQ1", "RQ2", "RQ3"]
    assert protocol.generation == {"concepts": "accepted", "criteria": "accepted", "facets": "accepted"}


# ---------- G1 ----------
def test_g1_opened_and_waiting(store):
    write_questions(store)
    ctx, fake = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    result = ProtocolStage(year_from=2018).run(ctx)
    assert isinstance(result, StageResult) and result.status == "waiting_human"
    assert result.artifact == "protocol" and result.review_items[0].gate == "G1"
    assert ctx.queue.has_open("G1")
    protocol = store.read_artifact("protocol", Protocol)
    assert protocol.criteria.year_from == 2018
    assert {q.source for q in protocol.search_plan.queries} == {"openalex", "arxiv", "crossref", "semanticscholar"}


def test_resume_while_open_does_not_call_llm_again(store):
    write_questions(store)
    ctx, fake = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    ProtocolStage().run(ctx)
    again = ProtocolStage().run(ctx)
    assert again.status == "waiting_human" and fake.calls == 3
    assert len(ctx.queue.open_items("G1")) == 1  # pas de doublon


def test_close_requires_justification_then_done(store):
    write_questions(store)
    ctx, fake = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    first = ProtocolStage().run(ctx)
    item_id = first.review_items[0].id
    with pytest.raises(Exception):
        ctx.queue.close_item(item_id, APPROVED, "moi", "")
    ctx.queue.close_item(item_id, APPROVED, "moi", "relu et validé")
    done = ProtocolStage().run(ctx)
    assert done.status == "done" and done.artifact == "protocol"
    assert fake.calls == 3
    # la décision est journalisée
    assert ctx.queue.decisions.read()[0].justification == "relu et validé"
    # un point tranché n'est pas redemandé
    assert ProtocolStage().run(ctx).status == "done"


def test_rejected_g1_fails_with_justification(store):
    write_questions(store)
    ctx, _ = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    item_id = ProtocolStage().run(ctx).review_items[0].id
    ctx.queue.close_item(item_id, "rejected", "moi", "critères trop larges")
    result = ProtocolStage().run(ctx)
    assert result.status == "failed" and "critères trop larges" in result.message


def test_human_edit_of_concepts_rebuilds_queries_on_approval(store):
    write_questions(store)
    ctx, _ = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    item_id = ProtocolStage(year_from=2020).run(ctx).review_items[0].id
    path = store.artifact_path("protocol")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["concepts"][1]["terms"].append("ChatGPT")
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx.queue.close_item(item_id, APPROVED, "moi", "ajout de ChatGPT")
    assert ProtocolStage().run(ctx).status == "done"
    protocol = store.read_artifact("protocol", Protocol)
    openalex = next(q for q in protocol.search_plan.queries if q.source == "openalex")
    assert "ChatGPT" in openalex.query
    assert openalex.params == {"filter": "from_publication_date:2020-01-01"}  # année conservée


def test_invalid_human_edit_is_refused_at_approval(store):
    write_questions(store)
    ctx, _ = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    item_id = ProtocolStage().run(ctx).review_items[0].id
    path = store.artifact_path("protocol")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["concepts"][0]["terms"] = ["x"]
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx.queue.close_item(item_id, APPROVED, "moi", "test")
    with pytest.raises(ArtifactInvalid):
        ProtocolStage().run(ctx)


# ---------- repair / échec ----------
def test_one_malformed_section_gets_exactly_one_repair(store):
    write_questions(store)
    ctx, fake = make_ctx(store, ["pas le bon format", CONCEPTS, CRITERIA, FACETS])
    result = ProtocolStage().run(ctx)
    assert result.status == "waiting_human" and fake.calls == 4
    assert store.read_artifact("protocol", Protocol).generation["concepts"] == "repaired"


def test_section_failing_after_repair_gives_failed_and_no_artifact(store):
    write_questions(store)
    ctx, fake = make_ctx(store, ["x", "y"])
    result = ProtocolStage().run(ctx)
    assert result.status == "failed" and "concepts" in result.message
    assert not store.exists("protocol") and not ctx.queue.has_open("G1")
    assert fake.calls == 2  # ni critères ni facettes demandés


def test_llm_failure_is_reported_not_raised(store):
    write_questions(store)
    ctx, fake = make_ctx(store, [])  # FakeLLM sans réponse => échec d'appel
    result = ProtocolStage().run(ctx)
    assert result.status == "failed"


def test_trace_is_written(store):
    write_questions(store)
    ctx, _ = make_ctx(store, [CONCEPTS, CRITERIA, FACETS])
    ProtocolStage().run(ctx)
    events = [e["event"] for e in ctx.trace.read()]
    assert "written" in events
    assert {"concepts_accepted", "criteria_accepted", "facets_accepted"} <= set(events)