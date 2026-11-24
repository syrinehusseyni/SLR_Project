import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.config import Settings
from slrresearch.core.contract import RunContext, Stage, StageResult, build_context
from slrresearch.core.errors import ConfigError
from slrresearch.core.schemas import Paper, SearchResult
from slrresearch.core.schemas.run import RunState, StageStatus
from slrresearch.hitl.queue import ReviewItem
from tests.fakes import FakeLLM


def make_item() -> ReviewItem:
    return ReviewItem(id="G1:protocol:p1", gate="G1", target_type="protocol", target_id="p1", reason="valider")


# -- StageResult : coherence ------------------------------------------------
def test_done_requires_artifact():
    with pytest.raises(ValueError, match="artefact"):
        StageResult(status="done")
    assert StageResult(status="done", artifact="search").artifact == "search"


def test_waiting_human_requires_review_items():
    with pytest.raises(ValueError, match="point de revue"):
        StageResult(status="waiting_human")
    assert StageResult(status="waiting_human", review_items=[make_item()]).status == "waiting_human"


def test_failed_requires_message():
    with pytest.raises(ValueError, match="message"):
        StageResult(status="failed")
    assert StageResult(status="failed", message="API 500").message == "API 500"


# -- RunState : schema + stockage ----------------------------------------------
def test_run_state_roundtrip_through_store(store):
    state = RunState(
        run_id="run_test",
        topic="LLM pour le screening",
        stages={"protocol": StageStatus(name="protocol", state="waiting_human")},
    )
    store.write_artifact("state", RunState, state)
    back = store.read_artifact("state", RunState)
    assert back.stages["protocol"].state == "waiting_human"


def test_invalid_stage_state_refused():
    with pytest.raises(ValueError):
        StageStatus(name="search", state="en_vacances")


# -- build_context : branchement ---------------------------------------------------
def test_build_context_wires_llm_by_role_with_shared_budget_and_trace(store):
    settings = Settings.from_env({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL_JUDGE": "mistral:7b"})
    budget = Budget(max_calls=10)
    made = []

    def fake_make_model(s, role):
        made.append(role)
        return FakeLLM(["reponse"] * 5)

    ctx = build_context(store, settings, budget=budget, make_model=fake_make_model)
    judge = ctx.llm("judge")
    screening = ctx.llm("screening")
    assert judge.model_name == "mistral:7b"  # modele du role "judge"
    assert screening.model_name == "llama3.1:8b"
    judge.call("p")
    screening.call("p")
    assert budget.calls == 2  # budget partage entre les roles
    assert [e["agent"] for e in ctx.trace.read()] == ["judge", "screening"]  # trace partagee
    assert made == ["judge", "screening"]


def test_llm_client_is_cached_per_role(store):
    made = []
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: made.append(r) or FakeLLM(["x"]))
    assert ctx.llm("writer") is ctx.llm("writer")
    assert made == ["writer"]  # le modele n'est construit qu'une fois


def test_context_without_factory_refuses_llm(store, trace):
    ctx = RunContext(store=store, trace=trace, queue=None, settings=Settings.from_env({}))
    with pytest.raises(ConfigError, match="llm_factory"):
        ctx.llm("screening")


# -- Une etape factice qui respecte le contrat ---------------------------------------------
class DummySearchStage:
    name = "search"

    def run(self, ctx: RunContext) -> StageResult:
        ctx.trace.log(self.name, "stage_started")
        result = SearchResult(query="llm screening", source="openalex",
                              papers=[Paper(id="W1", title="Un article", source="openalex")])
        ctx.store.write_artifact("search", SearchResult, result)
        item = ctx.queue.open_item("G2", "paper", "W1", "article sans resume")
        ctx.trace.log(self.name, "stage_finished", papers=1)
        return StageResult(status="waiting_human", artifact="search", review_items=[item])


def test_dummy_stage_follows_the_contract(store):
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: FakeLLM([]))
    stage = DummySearchStage()
    assert isinstance(stage, Stage)

    result = stage.run(ctx)

    assert result.status == "waiting_human"
    assert store.read_artifact(result.artifact, SearchResult).papers[0].id == "W1"  # artefact valide
    assert ctx.queue.has_open("G2")  # chemin humain
    assert [e["event"] for e in ctx.trace.read()] == ["stage_started", "stage_finished"]  # trace