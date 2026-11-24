import json

import pytest

from slrresearch.core.schemas.extraction import ExtractionOutcome, PaperText
from slrresearch.extraction import stage as stage_module
from slrresearch.extraction.reader import paper_file
from slrresearch.extraction.stage import ExtractionStage
from tests.extraction_helpers import make_pdf, prose, setup_documents


@pytest.fixture
def store(tmp_path):
    from slrresearch.core.store import RunStore
    return RunStore(tmp_path, "run1")


def read_index(store) -> ExtractionOutcome:
    return store.read_artifact("extraction", ExtractionOutcome)


def count_extractions(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = stage_module.extract_paper

    def spy(paper_id, data, params, backend="auto"):
        calls.append(paper_id)
        return real(paper_id, data, params, backend)

    monkeypatch.setattr(stage_module, "extract_paper", spy)
    return calls


def test_happy_path_writes_index_and_one_file_per_paper(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("alpha"), prose("beta")]), "B": make_pdf([prose("gamma")])})
    result = ExtractionStage(chunk_size=300, chunk_overlap=50).run(ctx)
    assert result.status == "done" and result.artifact == "extraction" and result.message == ""
    index = read_index(store)
    assert [p.paper_id for p in index.papers] == ["A", "B"]          # ordre de `documents`
    assert (index.summary.total, index.summary.extracted, index.summary.failed) == (2, 2, 0)
    assert index.summary.chunks == sum(p.chunks for p in index.papers) > 0
    for entry in index.papers:
        text = PaperText.model_validate_json((store.run_dir / entry.file).read_text(encoding="utf-8"))
        assert len(text.chunks) == entry.chunks and text.source_sha256 == entry.source_sha256
    assert index.limits.chunk_size == 300 and index.limits.max_papers is None


def test_provenance_a_quote_is_found_in_the_right_page(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("alpha"), prose("beta")])})
    ExtractionStage(chunk_size=300, chunk_overlap=50).run(ctx)
    text = PaperText.model_validate_json(paper_file(store.run_dir, "A").read_text(encoding="utf-8"))
    quote = "Sentence 3 of beta"
    flat = lambda s: " ".join(s.split())  # noqa: E731
    hits = [c for c in text.chunks if quote in flat(c.text)]
    assert hits and all(c.page == 2 for c in hits)


def test_unavailable_papers_are_counted_not_extracted(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("alpha")]), "U": "unavailable"})
    assert ExtractionStage().run(ctx).status == "done"
    index = read_index(store)
    assert [p.paper_id for p in index.papers] == ["A"] and index.summary.skipped_unavailable == 1


def test_missing_documents_artifact(store):
    from slrresearch.core.config import Settings
    from slrresearch.core.contract import build_context
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: None)
    result = ExtractionStage().run(ctx)
    assert result.status == "failed" and "documents" in result.message and "download" in result.message


def test_open_g3_blocks_the_stage(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "M": "manual_required"})
    result = ExtractionStage().run(ctx)
    assert result.status == "failed" and "G3" in result.message
    assert not store.exists("extraction") and not (store.run_dir / "extraction").exists()


def test_nothing_downloaded(store):
    ctx = setup_documents(store, {"U": "unavailable"})
    result = ExtractionStage().run(ctx)
    assert result.status == "failed" and "rien à extraire" in result.message


def test_resume_does_not_reread_up_to_date_papers(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")])})
    calls = count_extractions(monkeypatch)
    ExtractionStage().run(ctx)
    assert calls == ["A", "B"]
    calls.clear()
    assert ExtractionStage().run(ctx).status == "done"
    assert calls == []                                   # rien n'est relu
    assert read_index(store).summary.extracted == 2      # l'index est bien reconstruit depuis les fichiers


def test_force_redoes_everything(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")])})
    calls = count_extractions(monkeypatch)
    ExtractionStage().run(ctx)
    calls.clear()
    ExtractionStage(force=True).run(ctx)
    assert calls == ["A"]


def test_changed_params_redo_the_papers(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")])})
    calls = count_extractions(monkeypatch)
    ExtractionStage(chunk_size=400, chunk_overlap=50).run(ctx)
    calls.clear()
    ExtractionStage(chunk_size=400, chunk_overlap=50).run(ctx)
    assert calls == []
    ExtractionStage(chunk_size=500, chunk_overlap=50).run(ctx)
    assert calls == ["A"]
    ExtractionStage(chunk_size=500, chunk_overlap=50, max_pages=1).run(ctx)
    assert calls == ["A", "A"]


def test_corrupted_cache_file_is_redone(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")])})
    ExtractionStage().run(ctx)
    paper_file(store.run_dir, "A").write_text("{cassé", encoding="utf-8")
    calls = count_extractions(monkeypatch)
    assert ExtractionStage().run(ctx).status == "done" and calls == ["A"]


def test_replaced_pdf_is_detected_and_reported(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")])})
    (store.run_dir / "documents" / "A.pdf").write_bytes(make_pdf([prose("autre contenu")]))   # empreinte différente
    result = ExtractionStage().run(ctx)
    assert result.status == "done" and "A" in result.message and "changé" in result.message
    index = read_index(store)
    assert [(p.paper_id, p.status) for p in index.papers] == [("A", "failed"), ("B", "extracted")]
    assert not paper_file(store.run_dir, "A").exists()


def test_missing_pdf_file_is_a_failed_paper_not_a_crash(store):
    ctx = setup_documents(store, {"A": None, "B": make_pdf([prose("b")])})
    result = ExtractionStage().run(ctx)
    assert result.status == "done"
    entry = read_index(store).papers[0]
    assert entry.status == "failed" and "introuvable" in entry.reason


def test_scanned_pdf_fails_that_paper_only(store):
    ctx = setup_documents(store, {"S": make_pdf(["", ""]), "B": make_pdf([prose("b")])})
    result = ExtractionStage().run(ctx)
    assert result.status == "done" and "S" in result.message
    index = read_index(store)
    assert (index.summary.extracted, index.summary.failed) == (1, 1)
    assert "couche texte" in index.papers[0].reason


def test_all_papers_failing_gives_failed_but_still_writes_the_index(store):
    ctx = setup_documents(store, {"S": make_pdf(["", ""])})
    result = ExtractionStage().run(ctx)
    assert result.status == "failed" and "aucun papier" in result.message
    assert read_index(store).summary.failed == 1


def test_a_failed_paper_is_retried_after_a_fix(store):
    ctx = setup_documents(store, {"S": make_pdf(["", ""])})
    assert ExtractionStage().run(ctx).status == "failed"
    ctx = setup_documents(store, {"S": make_pdf([prose("enfin")])})   # le papier est remplacé (nouveau PDF, nouvelle empreinte)
    assert ExtractionStage().run(ctx).status == "done"


def test_max_papers_refuses_before_any_work(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")]), "C": make_pdf([prose("c")])})
    calls = count_extractions(monkeypatch)
    result = ExtractionStage(max_papers=2).run(ctx)
    assert result.status == "failed" and "--max-papers 2" in result.message
    assert calls == [] and not (store.run_dir / "extraction").exists() and not store.exists("extraction")


def test_max_papers_counts_only_papers_still_to_extract(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")])})
    ExtractionStage().run(ctx)
    assert ExtractionStage(max_papers=1).run(ctx).status == "done"   # tout est déjà extrait : rien à faire


def test_max_pages_is_recorded_and_flags_truncation(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("a"), prose("b"), prose("c")])})
    ExtractionStage(max_pages=2).run(ctx)
    index = read_index(store)
    entry = index.papers[0]
    assert (entry.total_pages, entry.pages_read, entry.truncated) == (3, 2, True) and index.limits.max_pages == 2


def test_ctrl_c_keeps_finished_papers_and_resume_finishes_the_rest(store, monkeypatch):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "B": make_pdf([prose("b")]), "C": make_pdf([prose("c")])})
    real = stage_module.extract_paper

    def interrupt_on_c(paper_id, data, params, backend="auto"):
        if paper_id == "C":
            raise KeyboardInterrupt
        return real(paper_id, data, params, backend)

    monkeypatch.setattr(stage_module, "extract_paper", interrupt_on_c)
    with pytest.raises(KeyboardInterrupt):
        ExtractionStage().run(ctx)
    assert paper_file(store.run_dir, "A").exists() and paper_file(store.run_dir, "B").exists()
    assert not paper_file(store.run_dir, "C").exists() and not store.exists("extraction")
    monkeypatch.setattr(stage_module, "extract_paper", real)
    calls = count_extractions(monkeypatch)
    assert ExtractionStage().run(ctx).status == "done"
    assert calls == ["C"]                                       # seul le papier manquant est lu
    assert read_index(store).summary.extracted == 3


def test_progress_callback_and_trace(store):
    ctx = setup_documents(store, {"A": make_pdf([prose("a")]), "S": make_pdf([""])})
    seen = []
    ExtractionStage(progress=lambda i, n, pid, status: seen.append((i, n, pid, status.split(":")[0].split(",")[0].strip()))).run(ctx)
    assert [s[:3] for s in seen] == [(1, 2, "A"), (2, 2, "S")] and seen[1][3] == "échec"
    events = [json.loads(line)["event"] for line in store.trace_path.read_text(encoding="utf-8").splitlines()]
    assert {"started", "paper_done", "paper_failed", "written"} <= set(events)


def test_colliding_file_names_are_refused(store):
    ctx = setup_documents(store, {"a/b": make_pdf([prose("a")]), "a b": make_pdf([prose("b")])})
    result = ExtractionStage().run(ctx)
    assert result.status == "failed" and "même nom" in result.message


def test_invalid_settings_fail_at_construction():
    with pytest.raises(ValueError):
        ExtractionStage(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        ExtractionStage(chunk_size=0)