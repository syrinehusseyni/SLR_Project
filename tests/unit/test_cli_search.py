import httpx

from cli import main
from tests.fakes import FakeLLM
from tests.unit.test_protocol_stage import CONCEPTS, CRITERIA, FACETS, Q_TEXT
from tests.unit.test_search_stage import Server


def test_search_through_cli(tmp_path, capsys):
    fake = FakeLLM([Q_TEXT, CONCEPTS, CRITERIA, FACETS])
    make = lambda s, r: fake
    root = ["--root", str(tmp_path)]
    env = {"OPENALEX_API_KEY": "k", "CONTACT_EMAIL": "me@example.org"}
    options = {"sleep": lambda s: None}
    transport = httpx.MockTransport(Server())

    assert main(root + ["questions", "--run-id", "d", "--topic", "LLM et screening"], make_model=make) == 0
    assert main(root + ["protocol", "--run-id", "d", "--year-from", "2018"], make_model=make) == 2
    # G1 pas encore approuvée : la recherche refuse
    assert main(root + ["search", "--run-id", "d"], make_model=make, transport=transport, env=env, stage_options=options) == 1
    assert "G1" in capsys.readouterr().out
    main(root + ["close", "--run-id", "d", "--item", "G1:protocol:main", "--decision", "approved", "--by", "m", "--why", "ok"], make_model=make)
    assert main(root + ["protocol", "--run-id", "d"], make_model=make) == 0

    capsys.readouterr()
    assert main(root + ["search", "--run-id", "d", "--limit", "20"], make_model=make, transport=transport, env=env, stage_options=options) == 0
    out = capsys.readouterr().out
    assert "Notices récupérées" in out and "papiers uniques" in out
    assert (tmp_path / "d" / "artifacts" / "search.json").exists()