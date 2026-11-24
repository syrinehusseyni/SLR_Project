import json

import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.errors import ConfigError, LLMError
from slrresearch.core.schemas.extraction import ExtractionOutcome
from slrresearch.core.schemas.study_card import StudyCard, StudyCardsOutcome
from slrresearch.extraction import card_stage as stage_module
from slrresearch.extraction.card_stage import StudyCardStage
from slrresearch.extraction.reader import paper_file
from slrresearch.extraction.stage import ExtractionStage
from slrresearch.extraction.study_card import card_file
from tests.extraction_helpers import make_pdf, prose, setup_documents
from tests.study_card_helpers import RuleLLM, make_ctx, prepare_run, rich_text


@pytest.fixture
def store(tmp_path):
    from slrresearch.core.store import RunStore
    return RunStore(tmp_path, "run1")


def index(store) -> StudyCardsOutcome:
    return store.read_artifact("study_cards", StudyCardsOutcome)


def read_card(store, paper_id) -> StudyCard:
    return StudyCard.model_validate_json(card_file(store.run_dir, paper_id).read_text(encoding="utf-8"))


def test_happy_path_cards_are_verified_with_pages(store):
    prepare_run(store)
    llm = RuleLLM()
    result = StudyCardStage().run(make_ctx(store, llm))
    assert result.status == "done" and result.artifact == "study_cards" and result.message == ""
    idx = index(store)
    assert [p.paper_id for p in idx.papers] == ["A", "B"]
    assert (idx.summary.total, idx.summary.complete, idx.summary.llm_calls) == (2, 2, 10) and llm.calls == 10
    assert idx.summary.fields == {"verified": 10}
    card = read_card(store, "A")
    assert [f.field_id for f in card.fields] == ["design", "data", "models", "findings", "limitations"]
    assert all(f.status == "verified" and f.page in (1, 2) and f.chunk_id.startswith("A_p") for f in card.fields)
    assert idx.limits.model == "llama3.1:8b" and idx.limits.top_k == 3 and idx.limits.prompt == "study_card.txt"


def test_resume_does_not_ask_again(store):
    prepare_run(store)
    llm = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm))
    llm2 = RuleLLM()
    result = StudyCardStage().run(make_ctx(store, llm2))
    assert result.status == "done" and llm2.calls == 0 and index(store).summary.llm_calls == 0
    assert index(store).summary.complete == 2


def test_force_redoes_everything(store):
    prepare_run(store)
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    llm = RuleLLM()
    StudyCardStage(force=True).run(make_ctx(store, llm))
    assert llm.calls == 10


def test_changing_the_model_or_top_k_invalidates_cards(store):
    prepare_run(store)
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    llm = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm, env={"OLLAMA_MODEL_EXTRACTION": "autre-modele"}))
    assert llm.calls == 10 and read_card(store, "A").model == "autre-modele"
    llm = RuleLLM()
    StudyCardStage(top_k=2).run(make_ctx(store, llm, env={"OLLAMA_MODEL_EXTRACTION": "autre-modele"}))
    assert llm.calls == 10


def test_changed_extraction_invalidates_cards(store):
    ctx = prepare_run(store)
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    ExtractionStage(chunk_size=700, chunk_overlap=80).run(ctx)       # autres chunks => autre empreinte
    llm = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm))
    assert llm.calls == 10


def test_only_failed_fields_are_retried_on_resume(store):
    prepare_run(store)
    state = {"failed": False}

    def flaky(prompt):
        if "Limites :" in prompt and not state["failed"]:
            state["failed"] = True
            raise LLMError("429 simulé")
        return RuleLLM.quote_first_chunk(prompt)

    llm = RuleLLM(flaky)
    result = StudyCardStage().run(make_ctx(store, llm))
    assert result.status == "done" and "1 fiches incomplètes" in result.message
    idx = index(store)
    assert [(p.paper_id, p.status) for p in idx.papers] == [("A", "incomplete"), ("B", "complete")]
    assert idx.papers[0].counts == {"verified": 4, "llm_error": 1}
    llm2 = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm2))
    assert llm2.calls == 1                                            # seul le champ en panne est redemandé
    assert index(store).summary.complete == 2 and index(store).summary.fields == {"verified": 10}


def test_rejected_and_not_reported_fields_are_kept_not_retried(store):
    prepare_run(store, tags=("A",))

    def rule(prompt):
        if "Type d'étude :" in prompt:
            chunk_id, _ = RuleLLM.first_chunk(prompt)
            return json.dumps({"status": "found", "value": "v", "chunk_id": chunk_id, "quote": "an invented sentence not in the paper"})
        if "Limites :" in prompt:
            return '{"status": "not_reported"}'
        return RuleLLM.quote_first_chunk(prompt)

    StudyCardStage().run(make_ctx(store, RuleLLM(rule)))
    card = read_card(store, "A")
    statuses = {f.field_id: f.status for f in card.fields}
    assert statuses["design"] == "rejected" and statuses["limitations"] == "not_reported" and statuses["data"] == "verified"
    assert index(store).papers[0].status == "complete"
    llm = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm))
    assert llm.calls == 0


def test_no_context_when_nothing_in_the_paper_looks_like_the_field(store):
    ctx = setup_documents(store, {"Z": make_pdf([prose("zeta")] * 2)})   # du texte sans design, sans modèle, sans limites...
    ExtractionStage(chunk_size=500, chunk_overlap=80).run(ctx)
    llm = RuleLLM()
    StudyCardStage().run(make_ctx(store, llm))
    statuses = {f.field_id: f.status for f in read_card(store, "Z").fields}
    assert statuses["findings"] == "verified" and "no_context" in statuses.values()
    assert llm.calls == sum(1 for s in statuses.values() if s != "no_context")   # aucun appel pour un champ sans contexte


def test_max_papers_refuses_before_any_call(store):
    prepare_run(store, tags=("A", "B", "C"))
    llm = RuleLLM()
    result = StudyCardStage(max_papers=2).run(make_ctx(store, llm))
    assert result.status == "failed" and "--max-papers 2" in result.message
    assert llm.calls == 0 and not (store.run_dir / "study_cards").exists() and not store.exists("study_cards")


def test_max_papers_counts_only_papers_still_to_process(store):
    prepare_run(store)
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert StudyCardStage(max_papers=1).run(make_ctx(store, RuleLLM())).status == "done"


def test_budget_stop_between_papers_keeps_done_cards_and_resumes(store):
    prepare_run(store)
    llm = RuleLLM()
    result = StudyCardStage().run(make_ctx(store, llm, budget=Budget(max_calls=7)))
    assert result.status == "failed" and "budget" in result.message and "--max-calls" in result.message
    assert llm.calls == 5 and card_file(store.run_dir, "A").exists() and not card_file(store.run_dir, "B").exists()
    assert not store.exists("study_cards")
    llm2 = RuleLLM()
    result = StudyCardStage().run(make_ctx(store, llm2, budget=Budget(max_calls=10)))
    assert result.status == "done" and llm2.calls == 5                  # seul le papier B est fait
    assert index(store).summary.complete == 2


def test_budget_exhausted_in_the_middle_of_a_paper(store):
    prepare_run(store, tags=("A",))
    llm = RuleLLM(lambda prompt: "pas du json")                          # chaque champ coûte 2 appels (essai + réparation)
    result = StudyCardStage().run(make_ctx(store, llm, budget=Budget(max_calls=5)))
    assert result.status == "failed" and "budget" in result.message and "interrompu" in result.message
    assert not card_file(store.run_dir, "A").exists() and not store.exists("study_cards")
    assert llm.calls == 5


def test_extraction_artifact_missing(store):
    result = StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert result.status == "failed" and "extraction" in result.message


def test_papers_not_extracted_are_skipped_and_counted(store):
    ctx = setup_documents(store, {"A": make_pdf([rich_text("A")] * 2), "S": make_pdf(["", ""])})
    ExtractionStage(chunk_size=500, chunk_overlap=80).run(ctx)
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    idx = index(store)
    assert [p.paper_id for p in idx.papers] == ["A"] and idx.summary.skipped_not_extracted == 1


def test_nothing_extracted_at_all(store):
    ctx = setup_documents(store, {"S": make_pdf(["", ""])})
    ExtractionStage().run(ctx)
    result = StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert result.status == "failed" and "rien à résumer" in result.message


def test_unreadable_extraction_file_fails_that_paper_only(store):
    prepare_run(store)
    paper_file(store.run_dir, "A").write_text("{cassé", encoding="utf-8")
    result = StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert result.status == "done" and "1 en échec" in result.message
    idx = index(store)
    assert [(p.paper_id, p.status) for p in idx.papers] == [("A", "failed"), ("B", "complete")]
    assert "illisible" in idx.papers[0].reason and not card_file(store.run_dir, "A").exists()


def test_all_papers_failing_is_failed_but_the_index_is_written(store):
    prepare_run(store, tags=("A",))
    paper_file(store.run_dir, "A").write_text("{cassé", encoding="utf-8")
    result = StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert result.status == "failed" and "aucune fiche complète" in result.message and index(store).summary.failed == 1


def test_ctrl_c_keeps_finished_cards(store, monkeypatch):
    prepare_run(store, tags=("A", "B", "C"))
    real = stage_module.fill_field

    def interrupt_on_c(client, spec, chunks, template, top_k):
        if chunks[0].paper_id == "C":
            raise KeyboardInterrupt
        return real(client, spec, chunks, template, top_k)

    monkeypatch.setattr(stage_module, "fill_field", interrupt_on_c)
    with pytest.raises(KeyboardInterrupt):
        StudyCardStage().run(make_ctx(store, RuleLLM()))
    assert card_file(store.run_dir, "A").exists() and card_file(store.run_dir, "B").exists()
    assert not card_file(store.run_dir, "C").exists() and not store.exists("study_cards")
    assert "interrupted" in [json.loads(line)["event"] for line in store.trace_path.read_text(encoding="utf-8").splitlines()]
    monkeypatch.setattr(stage_module, "fill_field", real)
    llm = RuleLLM()
    assert StudyCardStage().run(make_ctx(store, llm)).status == "done" and llm.calls == 5


def test_pause_between_calls_and_progress(store):
    prepare_run(store)
    naps, seen = [], []
    stage = StudyCardStage(pause=2.5, sleep=naps.append, progress=lambda i, n, pid, status: seen.append((i, n, pid)))
    stage.run(make_ctx(store, RuleLLM()))
    assert naps == [2.5] * 9 and seen == [(1, 2, "A"), (2, 2, "B")]


def test_trace_events(store):
    prepare_run(store, tags=("A",))
    StudyCardStage().run(make_ctx(store, RuleLLM()))
    events = [json.loads(line)["event"] for line in store.trace_path.read_text(encoding="utf-8").splitlines()]
    assert {"started", "field_done", "paper_done", "written"} <= set(events)


def test_missing_prompt_file_is_a_config_error(store, tmp_path):
    prepare_run(store, tags=("A",))
    with pytest.raises(ConfigError):
        StudyCardStage(prompt_dir=tmp_path).run(make_ctx(store, RuleLLM()))


def test_invalid_settings_fail_at_construction():
    for kwargs in ({"top_k": 0}, {"pause": -1}):
        with pytest.raises(ValueError):
            StudyCardStage(**kwargs)