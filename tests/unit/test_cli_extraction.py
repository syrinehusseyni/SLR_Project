"""Les commandes `extraction` et `studycards` de cli.py : arguments, budgets, codes de sortie, résumés affichés."""
import cli
from slrresearch.core.store import RunStore
from tests.extraction_helpers import make_pdf, prose, setup_documents
from tests.study_card_helpers import RuleLLM, prepare_run, rich_text


def run_cli(tmp_path, *argv, llm=None):
    llm = llm or RuleLLM()
    return cli.main(["--root", str(tmp_path), *argv], make_model=lambda settings, role: llm), llm


def store_for(tmp_path, run_id="r"):
    return RunStore(tmp_path, run_id)


def test_extraction_command_end_to_end(tmp_path, capsys):
    setup_documents(store_for(tmp_path), {"A": make_pdf([rich_text("A")] * 2), "B": make_pdf([prose("b")]), "U": "unavailable"})
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r")
    out = capsys.readouterr().out
    assert code == 0 and "OK : étape terminée, artefact « extraction »" in out
    assert "[1/2] A : 2 pages" in out and "[2/2] B : 1 pages" in out
    assert "extraits : 2" in out and "sans PDF (étape 4) : 1" in out and "chunks de 1200 caractères, chevauchement 200" in out


def test_extraction_options_are_applied_and_reported(tmp_path, capsys):
    setup_documents(store_for(tmp_path), {"A": make_pdf([rich_text("A")] * 3)})
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r", "--chunk-size", "400", "--chunk-overlap", "50", "--max-pages", "2")
    out = capsys.readouterr().out
    assert code == 0 and "tronqué (2/3 pages lues)" in out and "chunks de 400 caractères, chevauchement 50" in out and "pages lues : 2" in out


def test_extraction_invalid_chunk_settings_exit_1(tmp_path, capsys):
    setup_documents(store_for(tmp_path), {"A": make_pdf([prose("a")])})
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r", "--chunk-size", "100", "--chunk-overlap", "100")
    assert code == 1 and "ERREUR" in capsys.readouterr().out


def test_extraction_max_papers_refuses_with_exit_1(tmp_path, capsys):
    setup_documents(store_for(tmp_path), {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")])})
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r", "--max-papers", "1")
    assert code == 1 and "--max-papers 1" in capsys.readouterr().out and not store_for(tmp_path).exists("extraction")


def test_extraction_without_documents_exit_1(tmp_path, capsys):
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r")
    assert code == 1 and "ÉCHEC" in capsys.readouterr().out


def test_extraction_shows_failed_papers_and_resumes(tmp_path, capsys):
    setup_documents(store_for(tmp_path), {"A": make_pdf([prose("a")]), "S": make_pdf(["", ""])})
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r")
    out = capsys.readouterr().out
    assert code == 0 and "en échec : 1" in out and "! S : pas de couche texte" in out
    code, _ = run_cli(tmp_path, "extraction", "--run-id", "r")
    out = capsys.readouterr().out
    assert code == 0 and "[1/2] A" not in out and "[2/2] S : échec" in out   # A est déjà extrait ; S est retenté


def test_studycards_command_end_to_end(tmp_path, capsys):
    prepare_run(store_for(tmp_path))
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r")
    out = capsys.readouterr().out
    assert code == 0 and llm.calls == 10 and "OK : étape terminée, artefact « study_cards »" in out
    assert "[1/2] A" in out and "[2/2] B" in out and "fiches complètes : 2" in out
    assert "Champs par statut : {'verified': 10}" in out and "10 appels LLM" in out and "modèle : " in out


def test_studycards_budget_option_stops_and_resumes(tmp_path, capsys):
    prepare_run(store_for(tmp_path))
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--max-calls", "7")
    out = capsys.readouterr().out
    assert code == 1 and llm.calls == 5 and "budget" in out and "--max-calls" in out
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--max-calls", "10")
    assert code == 0 and llm.calls == 5 and "fiches complètes : 2" in capsys.readouterr().out


def test_studycards_token_budget_is_wired_to_the_budget(tmp_path, capsys):
    prepare_run(store_for(tmp_path), tags=("A",))

    class Tokens(RuleLLM):
        def invoke(self, prompt):
            from types import SimpleNamespace
            text = super().invoke(prompt)
            return SimpleNamespace(content=text, usage_metadata={"total_tokens": 100})

    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--max-tokens", "250", llm=Tokens())
    out = capsys.readouterr().out
    assert code == 1 and llm.calls == 3 and "tokens" in out        # 3 appels = 300 tokens >= 250 : arrêt


def test_studycards_options_max_papers_top_k_pause_force(tmp_path, capsys):
    prepare_run(store_for(tmp_path), tags=("A", "B", "C"))
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--max-papers", "2")
    assert code == 1 and llm.calls == 0 and "--max-papers 2" in capsys.readouterr().out
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--top-k", "2", "--force", "--pause", "0")
    assert code == 0 and llm.calls == 15 and "top-k : 2" in capsys.readouterr().out


def test_studycards_invalid_options_exit_1(tmp_path, capsys):
    prepare_run(store_for(tmp_path), tags=("A",))
    code, llm = run_cli(tmp_path, "studycards", "--run-id", "r", "--top-k", "0")
    assert code == 1 and llm.calls == 0 and "ERREUR" in capsys.readouterr().out


def test_studycards_without_extraction_exit_1(tmp_path, capsys):
    code, _ = run_cli(tmp_path, "studycards", "--run-id", "r")
    assert code == 1 and "extraction" in capsys.readouterr().out


def test_studycards_rejected_fields_get_a_warning(tmp_path, capsys):
    prepare_run(store_for(tmp_path), tags=("A",))
    import json

    def invented(prompt):
        chunk_id, _ = RuleLLM.first_chunk(prompt)
        return json.dumps({"status": "found", "value": "v", "chunk_id": chunk_id, "quote": "an invented sentence not in the paper"})

    code, _ = run_cli(tmp_path, "studycards", "--run-id", "r", llm=RuleLLM(invented))
    assert code == 0 and "ne sont pas des faits" in capsys.readouterr().out


def test_existing_commands_keep_their_default_budget(tmp_path):
    args = cli._parser().parse_args(["--root", str(tmp_path), "gates", "--run-id", "r"])
    assert getattr(args, "max_calls", 40) == 40 and getattr(args, "max_tokens", None) is None