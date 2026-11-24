"""Contrôles de l'étape 3 : pause, lots, limite de papiers, nouveaux essais, arrêt sur erreurs LLM, progression, Ctrl+C."""
from types import SimpleNamespace

import pytest

import cli
from slrresearch.core.contract import build_context
from slrresearch.core.errors import LLMError
from slrresearch.core.schemas.screening import LLMResults, ScreeningOutcome
from slrresearch.core.store import RunStore
from slrresearch.screening.stage import ScreeningStage
from tests.fakes import FakeLLM
from tests.screening_helpers import SETTINGS, TEMPLATE, make_paper, setup_run, verdict_json


def stage(**kw):
    return ScreeningStage(prompt_template=TEMPLATE, **kw)


def papers(n):
    return [make_paper(f"P{i}") for i in range(n)]


def new_ctx(store, responses):
    fake = FakeLLM(list(responses))
    return build_context(store, SETTINGS, make_model=lambda s, r: fake), fake


class Scripted:
    """Faux client LLM : exécute un script (réponse texte, ou exception à lever) appel après appel."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def call(self, prompt):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return SimpleNamespace(text=step)


def test_pause_is_applied_between_new_calls_only(store):
    sleeps = []
    ctx, _ = setup_run(store, papers(3), llm_responses=[verdict_json()] * 3)
    stage(pause=2.0, sleep=sleeps.append).run(ctx)
    assert sleeps == [2.0, 2.0]                      # pas avant le premier appel


def test_no_pause_for_papers_that_cost_no_call(store):
    sleeps = []
    ctx, _ = setup_run(store, [make_paper("OLD", year=2010), make_paper("NOABS", abstract=""), make_paper("A")],
                       llm_responses=[verdict_json()])
    stage(pause=2.0, sleep=sleeps.append).run(ctx)
    assert sleeps == []


def test_batch_stops_cleanly_and_resumes_until_done(store):
    # 1 papier trop ancien (sans appel) + 5 papiers à passer au LLM : lots de 2 + 2 + 1
    ctx, fake = setup_run(store, [make_paper("OLD", year=2010)] + papers(5), llm_responses=[verdict_json()] * 2)
    r = stage(batch=2).run(ctx)
    assert r.status == "failed" and "lot de 2" in r.message and not store.exists("screening")
    assert len(store.read_artifact("screening_llm", LLMResults).results) == 2   # OLD ne compte pas dans le lot
    ctx2, fake2 = new_ctx(store, [verdict_json()] * 2)
    r = stage(batch=2).run(ctx2)
    assert r.status == "failed" and fake2.calls == 2
    ctx3, fake3 = new_ctx(store, [verdict_json()])
    assert stage(batch=2).run(ctx3).status == "done" and fake3.calls == 1       # dernier papier : terminé
    assert store.read_artifact("screening", ScreeningOutcome).summary.total == 6


def test_error_streak_stops_the_run_without_flooding_g2(store):
    ctx, _ = setup_run(store, papers(6))
    down = Scripted(*[LLMError("429")] * 6)
    ctx.llm_factory = lambda role: down
    r = stage(max_consecutive_errors=3).run(ctx)
    assert r.status == "failed" and "indisponible" in r.message and "--pause" in r.message
    assert down.calls == 3                                  # on n'insiste pas sur les 3 papiers restants
    assert not store.exists("screening") and not ctx.queue.open_items("G2")


def test_isolated_errors_do_not_stop_the_run(store):
    ctx, _ = setup_run(store, papers(4))
    ok = verdict_json()
    alt = Scripted(LLMError("429"), ok, LLMError("429"), ok)
    ctx.llm_factory = lambda role: alt
    r = stage(max_consecutive_errors=2).run(ctx)
    assert r.status == "waiting_human"                      # 2 papiers en erreur -> G2, les 2 autres décidés
    out = store.read_artifact("screening", ScreeningOutcome)
    assert out.summary.llm_failures == 2


def test_progress_is_reported_for_every_paper(store):
    seen = []
    ctx, _ = setup_run(store, [make_paper("OLD", year=2010), make_paper("A")], llm_responses=[verdict_json()])
    stage(progress=lambda i, n, pid, status: seen.append((i, n, pid, status))).run(ctx)
    assert seen == [(1, 2, "OLD", "not_run"), (2, 2, "A", "ok")]


def test_ctrl_c_saves_progress_then_reraises(store):
    ctx, _ = setup_run(store, papers(4))
    interrupted = Scripted(verdict_json(), verdict_json(), KeyboardInterrupt())
    ctx.llm_factory = lambda role: interrupted
    with pytest.raises(KeyboardInterrupt):
        stage(save_every=10).run(ctx)                       # sauvegarde périodique pas encore atteinte
    assert len(store.read_artifact("screening_llm", LLMResults).results) == 2
    ctx2, fake2 = new_ctx(store, [verdict_json()] * 2)
    assert stage().run(ctx2).status == "done" and fake2.calls == 2


def test_max_papers_refuses_before_any_llm_call(store):
    ctx, fake = setup_run(store, papers(5), llm_responses=[verdict_json()] * 5)
    r = stage(max_papers=3).run(ctx)
    assert r.status == "failed" and "5 papiers" in r.message and "--max-papers 3" in r.message
    assert "search" in r.message and fake.calls == 0
    assert not store.exists("screening") and not store.exists("screening_llm")


def test_max_papers_counts_only_papers_that_need_the_llm(store):
    # 1 trop ancien + 1 sans résumé + 3 normaux : seuls les 3 normaux comptent
    ctx, fake = setup_run(store, [make_paper("OLD", year=2010), make_paper("NOABS", abstract="")] + papers(3),
                          llm_responses=[verdict_json()] * 3)
    assert stage(max_papers=3).run(ctx).status == "waiting_human" and fake.calls == 3


def test_max_papers_ignores_papers_already_in_cache(store):
    ctx, _ = setup_run(store, papers(4), llm_responses=[verdict_json()] * 2)
    stage(batch=2).run(ctx)                                  # 2 papiers en cache
    ctx2, fake2 = new_ctx(store, [verdict_json()] * 2)
    assert stage(max_papers=2).run(ctx2).status == "done" and fake2.calls == 2


def test_failed_call_is_retried_after_a_wait(store):
    sleeps, seen = [], []
    ctx, _ = setup_run(store, papers(1))
    llm = Scripted(LLMError("429"), verdict_json())
    ctx.llm_factory = lambda role: llm
    r = stage(retries=2, retry_wait=5.0, sleep=sleeps.append,
              progress=lambda i, n, pid, status: seen.append(status)).run(ctx)
    assert r.status == "done" and llm.calls == 2 and sleeps == [5.0]
    assert any("nouvel essai 1/2 dans 5s" in x for x in seen) and seen[-1] == "ok"


def test_wait_grows_with_each_retry(store):
    sleeps = []
    ctx, _ = setup_run(store, papers(1))
    llm = Scripted(LLMError("429"), LLMError("429"), verdict_json())
    ctx.llm_factory = lambda role: llm
    assert stage(retries=2, retry_wait=5.0, sleep=sleeps.append).run(ctx).status == "done"
    assert sleeps == [5.0, 10.0]


def test_retries_exhausted_sends_the_paper_to_human_review(store):
    ctx, _ = setup_run(store, papers(1))
    llm = Scripted(LLMError("429"), LLMError("429"), LLMError("429"))
    ctx.llm_factory = lambda role: llm
    r = stage(retries=2, retry_wait=1.0, sleep=lambda s: None).run(ctx)
    assert r.status == "waiting_human" and llm.calls == 3


def test_rate_limit_blips_do_not_stop_a_long_run(store):
    ok = verdict_json()
    ctx, _ = setup_run(store, papers(4))
    llm = Scripted(LLMError("429"), ok, LLMError("429"), ok, LLMError("429"), ok, ok)   # 3 « coups » de limite de débit
    ctx.llm_factory = lambda role: llm
    r = stage(retries=1, retry_wait=1.0, max_consecutive_errors=2, sleep=lambda s: None).run(ctx)
    assert r.status == "done" and llm.calls == 7


# -- CLI -----------------------------------------------------------------------
def test_cli_batch_pause_and_progress_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    setup_run(RunStore(tmp_path, "r1"), papers(3), llm_responses=[])
    fake = FakeLLM([verdict_json()] * 3)
    code = cli.main(["--root", str(tmp_path), "screening", "--run-id", "r1", "--batch", "1", "--pause", "0"],
                    make_model=lambda s, r: fake)
    out = capsys.readouterr().out
    assert code == 1 and "[1/3] P0 : ok" in out and "lot de 1" in out and fake.calls == 1


def test_cli_ctrl_c_message_and_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    setup_run(RunStore(tmp_path, "r1"), papers(2), llm_responses=[])

    class Interrupting:
        def invoke(self, prompt):
            raise KeyboardInterrupt()

    code = cli.main(["--root", str(tmp_path), "screening", "--run-id", "r1"], make_model=lambda s, r: Interrupting())
    assert code == 130 and "INTERROMPU" in capsys.readouterr().out


def test_cli_max_papers_refuses_without_calling_the_llm(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    setup_run(RunStore(tmp_path, "r1"), papers(3), llm_responses=[])
    fake = FakeLLM([verdict_json()] * 3)
    code = cli.main(["--root", str(tmp_path), "screening", "--run-id", "r1", "--max-papers", "2"],
                    make_model=lambda s, r: fake)
    assert code == 1 and "--max-papers 2" in capsys.readouterr().out and fake.calls == 0