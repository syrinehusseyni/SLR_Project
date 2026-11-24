from slrresearch.core.budgets import Budget
from slrresearch.core.schemas.screening import LLMResults, ScreeningOutcome
from slrresearch.screening.stage import ScreeningStage
from tests.screening_helpers import (
    TEMPLATE, make_paper, make_protocol, setup_run, verdict_json,
)


def stage(**kw):
    return ScreeningStage(prompt_template=TEMPLATE, **kw)


def five_papers():
    return [
        make_paper("OLD", year=2010),                                   # exclu par l'année, sans LLM
        make_paper("NOABS", abstract=""),                               # sans résumé, sans LLM
        make_paper("OK"),                                               # include / include -> accord
        make_paper("DISAGREE", title="LLM title screening", abstract="A large language model for title screening."),
        make_paper("NONE", title="Soil study", abstract="We study soil chemistry."),  # A exclut, LLM exclut
    ]


def responses_for_three():
    return [
        verdict_json("include", 0.9),                                           # OK
        verdict_json("exclude", 0.9, evidence="large language model for title screening"),  # DISAGREE (A=include)
        verdict_json("exclude", 0.9, criteria=("I1",), evidence="We study soil chemistry"),  # NONE
    ]


def test_requires_protocol_search_and_g1(store):
    from slrresearch.core.config import Settings
    from slrresearch.core.contract import build_context
    from tests.fakes import FakeLLM
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: FakeLLM([]))
    assert "protocol" in stage().run(ctx).message
    ctx, fake = setup_run(store, five_papers(), approve_g1=False)
    r = stage().run(ctx)
    assert r.status == "failed" and "G1" in r.message and fake.calls == 0


def test_requires_search_artifact(store):
    ctx, _ = setup_run(store, five_papers())
    store.artifact_path("search").unlink()
    assert "search" in stage().run(ctx).message


def test_full_run_routes_each_paper_correctly(store):
    ctx, fake = setup_run(store, five_papers(), llm_responses=responses_for_three())
    r = stage().run(ctx)
    assert r.status == "waiting_human"
    assert fake.calls == 3                                    # OLD et NOABS n'ont pas coûté d'appel
    out = store.read_artifact("screening", ScreeningOutcome)
    rec = {x.paper_id: x for x in out.records}
    assert (rec["OLD"].final, rec["OLD"].origin) == ("exclude", "rule")
    assert rec["NOABS"].origin == "pending" and rec["NOABS"].llm_status == "not_run"
    assert (rec["OK"].final, rec["OK"].origin) == ("include", "agreement")
    assert rec["DISAGREE"].origin == "pending" and "Désaccord" in rec["DISAGREE"].escalation
    assert (rec["NONE"].final, rec["NONE"].origin) == ("exclude", "agreement")
    assert {i.id for i in r.review_items} == {"G2:paper:NOABS", "G2:paper:DISAGREE"}
    s = out.summary
    assert (s.total, s.included, s.excluded, s.pending, s.without_abstract) == (5, 1, 2, 2, 1)
    assert s.agreement.n == 3 and s.agreement.agreed == 2


def test_resume_applies_human_decisions_without_new_llm_calls(store):
    ctx, fake = setup_run(store, five_papers(), llm_responses=responses_for_three())
    stage().run(ctx)
    ctx.queue.close_item("G2:paper:NOABS", "include", "moi", "titre pertinent")
    assert stage().run(ctx).status == "waiting_human"        # il reste DISAGREE
    ctx.queue.close_item("G2:paper:DISAGREE", "exclude", "moi", "pas empirique")
    r = stage().run(ctx)
    assert r.status == "done" and r.artifact == "screening" and fake.calls == 3
    out = store.read_artifact("screening", ScreeningOutcome)
    assert all(x.final != "uncertain" for x in out.records) and out.summary.pending == 0
    human = {x.paper_id: x for x in out.records if x.origin == "human"}
    assert human["NOABS"].final == "include" and human["NOABS"].human_why == "titre pertinent"
    assert human["DISAGREE"].final == "exclude"


def test_invalid_human_decision_is_reported(store):
    ctx, _ = setup_run(store, [make_paper("NOABS", abstract="")])
    stage().run(ctx)
    ctx.queue.close_item("G2:paper:NOABS", "approved", "moi", "oups")   # le CLI l'empêche, le stage se protège
    r = stage().run(ctx)
    assert r.status == "failed" and "include ou exclude" in r.message


def test_budget_exhaustion_keeps_progress_and_resumes(store):
    papers = [make_paper(f"P{i}") for i in range(4)]
    ctx, fake = setup_run(store, papers, llm_responses=[verdict_json()] * 4, budget=Budget(max_calls=2))
    r = stage().run(ctx)
    assert r.status == "failed" and "budget" in r.message and not store.exists("screening")
    assert len(store.read_artifact("screening_llm", LLMResults).results) == 2
    # relance avec un nouveau contexte : seuls les 2 papiers restants appellent le LLM
    ctx2, fake2 = setup_run_again(store, [verdict_json()] * 2)
    assert stage().run(ctx2).status == "done" and fake2.calls == 2


def setup_run_again(store, responses):
    from slrresearch.core.contract import build_context
    from tests.fakes import FakeLLM
    from tests.screening_helpers import SETTINGS
    fake = FakeLLM(responses)
    return build_context(store, SETTINGS, make_model=lambda s, r: fake), fake


def test_llm_error_is_not_cached_and_is_retried(store):
    from slrresearch.core.errors import LLMError

    class Down:
        def call(self, prompt):
            raise LLMError("hors ligne")

    ctx, _ = setup_run(store, [make_paper("A")])
    ctx.llm_factory = lambda role: Down()
    r = stage().run(ctx)
    assert r.status == "waiting_human"
    out = store.read_artifact("screening", ScreeningOutcome)
    assert out.records[0].llm_status == "llm_error" and out.summary.llm_failures == 1
    assert store.read_artifact("screening_llm", LLMResults).results == {}   # pas de cache : sera retenté


def test_changed_criteria_invalidate_the_cache(store):
    ctx, fake = setup_run(store, [make_paper("A")], llm_responses=[verdict_json(), verdict_json()])
    stage().run(ctx)
    assert fake.calls == 1
    store.write_artifact("protocol", type(make_protocol()), make_protocol(year_from=2020).model_dump())
    assert stage(force=True).run(ctx).status in ("done", "waiting_human") and fake.calls == 2


def test_force_recomputes_but_reuses_llm_cache(store):
    ctx, fake = setup_run(store, [make_paper("A")], llm_responses=[verdict_json()])
    stage().run(ctx)
    r = stage(force=True).run(ctx)
    assert r.status == "done" and fake.calls == 1


def test_trace_records_the_run(store):
    ctx, _ = setup_run(store, five_papers(), llm_responses=responses_for_three())
    stage().run(ctx)
    events = {e["event"] for e in ctx.trace.read()}
    assert {"started", "llm_verdict", "written"} <= events


def test_stage_works_on_a_completely_different_subject(store):
    from slrresearch.core.schemas.protocol_plan import SearchConcept
    protocol = make_protocol(topic="Métformine et diabète", concepts=[
        SearchConcept(id="C1", label="médicament", terms=["metformin", "biguanide"]),
        SearchConcept(id="C2", label="maladie", terms=["type 2 diabetes", "glycemic control"])])
    paper = make_paper("M1", title="Metformin in type 2 diabetes", abstract="A randomised trial of metformin in type 2 diabetes.")
    ctx, fake = setup_run(store, [paper], protocol=protocol,
                          llm_responses=[verdict_json(evidence="randomised trial of metformin in type 2 diabetes")])
    assert stage().run(ctx).status == "done"
    assert "Métformine et diabète" in fake.prompts[0] and "LLM" not in fake.prompts[0]