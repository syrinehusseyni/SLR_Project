import json

from cli import main
from tests.fakes import FakeLLM
from tests.unit.test_protocol_stage import CONCEPTS, CRITERIA, FACETS, Q_TEXT


def test_full_flow_through_cli(tmp_path, capsys):
    fake = FakeLLM([Q_TEXT, CONCEPTS, CRITERIA, FACETS])
    make = lambda settings, role: fake
    root = ["--root", str(tmp_path)]

    assert main(root + ["questions", "--run-id", "demo", "--topic", "LLM et screening"], make_model=make) == 0
    assert (tmp_path / "demo" / "artifacts" / "questions.json").exists()

    assert main(root + ["protocol", "--run-id", "demo", "--year-from", "2018"], make_model=make) == 2
    assert (tmp_path / "demo" / "artifacts" / "protocol.json").exists()

    capsys.readouterr()
    assert main(root + ["gates", "--run-id", "demo"], make_model=make) == 0
    assert "G1:protocol:main" in capsys.readouterr().out

    assert main(root + ["close", "--run-id", "demo", "--item", "G1:protocol:main", "--decision", "approved",
                        "--by", "moi", "--why", "relu"], make_model=make) == 0
    assert main(root + ["protocol", "--run-id", "demo"], make_model=make) == 0

    capsys.readouterr()
    main(root + ["gates", "--run-id", "demo"], make_model=make)
    assert "Aucun point" in capsys.readouterr().out
    lines = (tmp_path / "demo" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["justification"] == "relu"


def test_close_without_known_item_returns_error(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "close", "--run-id", "demo", "--item", "G1:x:y",
                 "--decision", "approved", "--by", "m", "--why", "w"], make_model=lambda s, r: FakeLLM([]))
    assert code == 1 and "ERREUR" in capsys.readouterr().out