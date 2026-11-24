from pathlib import Path

import httpx

import cli
from slrresearch.core.store import RunStore
from tests.download_stage_helpers import paper, setup_run
from tests.fakes import FakeLLM
from tests.pdf_helpers import good_pdf

GOOD = good_pdf()


def handler(request):
    if "good.test" in str(request.url):
        return httpx.Response(200, content=GOOD, headers={"content-type": "application/pdf"})
    return httpx.Response(200, content=b"<html>login</html>")


def prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    store = RunStore(tmp_path, "r1")
    setup_run(store, [paper("A", pdf_url="https://good.test/a.pdf"), paper("B", pdf_url="https://paywall.test/b.pdf")],
              ["include", "include"])
    return store


def run(tmp_path, *args, email=""):
    return cli.main(["--root", str(tmp_path), *args], make_model=lambda s, r: FakeLLM([]),
                    transport=httpx.MockTransport(handler), env={"CONTACT_EMAIL": email},
                    stage_options={"sleep": lambda s: None})


# ---------------------------------------------------------------- défaut : papier sans PDF libre => indisponible, sans humain
def test_default_policy_declares_missing_papers_unavailable_without_any_human_step(tmp_path, monkeypatch, capsys):
    store = prepare(tmp_path, monkeypatch)
    assert run(tmp_path, "download", "--run-id", "r1", "--pause", "0") == 0
    out = capsys.readouterr().out
    assert "OK : étape terminée" in out and "PDF obtenus : 1" in out and "indisponibles : 1" in out
    assert "Dont 1 déclarés indisponibles automatiquement" in out and "on-missing=unavailable" in out
    assert run(tmp_path, "gates", "--run-id", "r1") == 0
    assert "Aucun point de revue ouvert" in capsys.readouterr().out
    assert store.exists("documents")


def test_review_option_keeps_the_g3_gate(tmp_path, monkeypatch, capsys):
    prepare(tmp_path, monkeypatch)
    assert run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0") == 2
    out = capsys.readouterr().out
    assert "G3:paper:B" in out and "on-missing=review" in out


def test_invalid_on_missing_value_is_rejected_by_the_parser(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    try:
        run(tmp_path, "download", "--run-id", "r1", "--on-missing", "ignore")
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("une valeur inconnue aurait dû être refusée")


# ---------------------------------------------------------------- mode review : porte G3
def test_download_waits_for_g3_then_completes(tmp_path, monkeypatch, capsys):
    store = prepare(tmp_path, monkeypatch)
    assert run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0") == 2
    out = capsys.readouterr().out
    assert "G3:paper:B" in out and "PDF obtenus : 1" in out and "CONTACT_EMAIL absent" in out
    assert "max-mb=50" in out and "provided" in out
    assert run(tmp_path, "gates", "--run-id", "r1") == 0
    assert "G3:paper:B" in capsys.readouterr().out
    assert run(tmp_path, "close", "--run-id", "r1", "--item", "G3:paper:B",
               "--decision", "Unavailable", "--by", "moi", "--why", "introuvable") == 0
    assert run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0") == 0
    assert "indisponibles : 1" in capsys.readouterr().out
    assert store.exists("documents")


def test_close_g3_refuses_invalid_word_and_keeps_item_open(tmp_path, monkeypatch, capsys):
    prepare(tmp_path, monkeypatch)
    run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0")
    capsys.readouterr()
    assert run(tmp_path, "close", "--run-id", "r1", "--item", "G3:paper:B",
               "--decision", "approved", "--by", "moi", "--why", "x") == 1
    assert "provided" in capsys.readouterr().out
    assert run(tmp_path, "gates", "--run-id", "r1") == 0
    assert "G3:paper:B" in capsys.readouterr().out


def test_close_g3_provided_checks_the_dropped_file_before_closing(tmp_path, monkeypatch, capsys):
    store = prepare(tmp_path, monkeypatch)
    run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0")
    capsys.readouterr()
    args = ("close", "--run-id", "r1", "--item", "G3:paper:B", "--decision", "provided", "--by", "moi", "--why", "déposé")
    assert run(tmp_path, *args) == 1 and "aucun fichier" in capsys.readouterr().out   # rien n'est fermé
    target = store.run_dir / "documents" / "manual" / "B.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(GOOD)
    assert run(tmp_path, *args) == 0
    assert run(tmp_path, "download", "--run-id", "r1", "--on-missing", "review", "--pause", "0") == 0
    assert "PDF obtenus : 2" in capsys.readouterr().out


def test_max_papers_option_refuses_cleanly(tmp_path, monkeypatch, capsys):
    prepare(tmp_path, monkeypatch)
    assert run(tmp_path, "download", "--run-id", "r1", "--max-papers", "1", "--pause", "0") == 1
    assert "--max-papers 1" in capsys.readouterr().out


def test_download_without_screening_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert run(tmp_path, "download", "--run-id", "vide") == 1
    assert "screening" in capsys.readouterr().out