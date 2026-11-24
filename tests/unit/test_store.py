import json

import pytest

from slrresearch.core.errors import ArtifactInvalid, ArtifactNotFound
from slrresearch.core.schemas import DocumentRecord, Paper, SearchResult
from slrresearch.core.store import RunStore


def make_result() -> SearchResult:
    return SearchResult(
        query="llm screening",
        source="openalex",
        papers=[Paper(id="W1", title="  LLMs   for screening ", source="openalex",
                      doi="https://doi.org/10.1000/ABC", year=2024)],
    )


def test_write_then_read_roundtrip(store):
    store.write_artifact("search_openalex", SearchResult, make_result())
    back = store.read_artifact("search_openalex", SearchResult)
    assert back.papers[0].title == "LLMs for screening"  # espaces normalises
    assert back.papers[0].doi == "10.1000/abc"  # DOI normalise


def test_invalid_artifact_is_refused_and_not_written(store):
    bad = {"query": "x", "source": "openalex", "papers": [{"id": "W1", "title": "", "source": "openalex"}]}
    with pytest.raises(ArtifactInvalid):
        store.write_artifact("bad", SearchResult, bad)
    assert not store.exists("bad")


def test_corrupted_file_detected_on_read(store):
    store.write_artifact("search_openalex", SearchResult, make_result())
    path = store.artifact_path("search_openalex")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["query"]  # quelqu'un a casse le fichier a la main
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactInvalid, match="corrompu"):
        store.read_artifact("search_openalex", SearchResult)


def test_missing_artifact_raises_not_found(store):
    with pytest.raises(ArtifactNotFound):
        store.read_artifact("nope", SearchResult)


def test_overwrite_keeps_backup(store):
    store.write_artifact("r", SearchResult, make_result())
    second = make_result()
    second.query = "autre requete"
    store.write_artifact("r", SearchResult, second)
    bak = store.artifact_path("r").with_suffix(".json.bak")
    assert bak.exists()
    assert json.loads(bak.read_text(encoding="utf-8"))["query"] == "llm screening"


def test_no_temp_file_left_behind(store):
    store.write_artifact("r", SearchResult, make_result())
    assert not list(store.artifacts_dir.glob("*.tmp"))


def test_path_traversal_refused(store, tmp_path):
    with pytest.raises(ValueError):
        store.artifact_path("../evil")
    with pytest.raises(ValueError):
        RunStore(tmp_path, "../evil")


def test_downloaded_document_requires_file_and_hash(store):
    with pytest.raises(ValueError):
        DocumentRecord(paper_id="W1", status="downloaded")
    ok = DocumentRecord(paper_id="W1", status="downloaded", pdf_path="x.pdf", sha256="abc")
    assert ok.status == "downloaded"
    assert DocumentRecord(paper_id="W1", status="manual_required").pdf_path == ""