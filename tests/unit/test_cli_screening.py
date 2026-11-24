from pathlib import Path

import cli
from slrresearch.core.contract import build_context
from slrresearch.core.store import RunStore
from tests.fakes import FakeLLM
from tests.screening_helpers import SETTINGS, make_paper, setup_run, verdict_json


def prepare(tmp_path, monkeypatch, papers, responses):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    store = RunStore(tmp_path, "r1")
    setup_run(store, papers, llm_responses=[])
    fake = FakeLLM(list(responses))
    return fake


def run(tmp_path, fake, *args):
    return cli.main(["--root", str(tmp_path), *args], make_model=lambda s, r: fake)


def test_screening_command_waits_then_completes(tmp_path, monkeypatch, capsys):
    fake = prepare(tmp_path, monkeypatch, [make_paper("A"), make_paper("NOABS", abstract="")], [verdict_json()])
    assert run(tmp_path, fake, "screening", "--run-id", "r1") == 2
    out = capsys.readouterr().out
    assert "G2:paper:NOABS" in out and "kappa" in out and "en attente (G2) : 1" in out
    assert run(tmp_path, fake, "gates", "--run-id", "r1") == 0
    assert "G2:paper:NOABS" in capsys.readouterr().out
    assert run(tmp_path, fake, "close", "--run-id", "r1", "--item", "G2:paper:NOABS",
               "--decision", "Include", "--by", "moi", "--why", "ok") == 0
    assert run(tmp_path, fake, "screening", "--run-id", "r1") == 0
    assert fake.calls == 1


def test_close_refuses_invalid_g2_decision_and_keeps_item_open(tmp_path, monkeypatch, capsys):
    fake = prepare(tmp_path, monkeypatch, [make_paper("NOABS", abstract="")], [])
    run(tmp_path, fake, "screening", "--run-id", "r1")
    capsys.readouterr()
    assert run(tmp_path, fake, "close", "--run-id", "r1", "--item", "G2:paper:NOABS",
               "--decision", "approved", "--by", "moi", "--why", "x") == 1
    assert "include" in capsys.readouterr().out
    ctx = build_context(RunStore(tmp_path, "r1"), SETTINGS, make_model=lambda s, r: fake)
    assert ctx.queue.get("G2:paper:NOABS").status == "open"


def test_screening_without_protocol_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path); monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert cli.main(["--root", str(tmp_path), "screening", "--run-id", "vide"], make_model=lambda s, r: FakeLLM([])) == 1
    assert "protocol" in capsys.readouterr().out