import json

import pytest

from slrresearch.core.schemas.extraction import ExtractionParams
from slrresearch.extraction.reader import ExtractionError, extract_paper, load_paper_text, paper_file, save_paper_text
from tests.extraction_helpers import make_pdf, prose

PARAMS = ExtractionParams(chunk_size=300, chunk_overlap=50, max_pages=None, min_chars_per_page=50, cleaning="1")


def test_extract_paper_reads_all_pages_with_provenance():
    data = make_pdf([prose("alpha"), prose("beta"), prose("gamma")])
    text = extract_paper("W1", data, PARAMS)
    assert (text.total_pages, text.pages_read, text.truncated) == (3, 3, False)
    assert len(text.page_chars) == 3 and all(n > 100 for n in text.page_chars)
    assert {c.page for c in text.chunks} == {1, 2, 3}
    page2 = [c for c in text.chunks if c.page == 2]
    assert page2 and all("beta" in c.text or "Sentence" in c.text for c in page2)
    assert any("of gamma" in c.text for c in text.chunks if c.page == 3)
    assert text.source_sha256 and text.paper_id == "W1"


def test_max_pages_truncates_but_keeps_real_total():
    data = make_pdf([prose("a"), prose("b"), prose("c")])
    text = extract_paper("W1", data, PARAMS.model_copy(update={"max_pages": 2}))
    assert (text.total_pages, text.pages_read, text.truncated) == (3, 2, True)
    assert max(c.page for c in text.chunks) == 2


def test_empty_pages_are_counted_not_fatal():
    text = extract_paper("W1", make_pdf([prose("a"), "", prose("c")]), PARAMS)
    assert text.page_chars[1] == 0 and 2 not in {c.page for c in text.chunks}


def test_pdf_without_text_layer_is_refused():
    with pytest.raises(ExtractionError, match="couche texte"):
        extract_paper("W1", make_pdf(["", "", ""]), PARAMS)


def test_unreadable_pdf_is_refused():
    with pytest.raises(ExtractionError, match="illisible"):
        extract_paper("W1", b"%PDF-1.4 ceci n'est pas un vrai PDF", PARAMS)
    with pytest.raises(ExtractionError):
        extract_paper("W1", b"pas du tout un pdf", PARAMS)


def test_both_backends_give_usable_text():
    data = make_pdf([prose("alpha")])
    for backend in ("pymupdf", "pypdf"):
        text = extract_paper("W1", data, PARAMS, backend=backend)
        assert any("Sentence 1 of alpha" in c.text.replace("\n", " ") for c in text.chunks), backend


def test_save_and_load_roundtrip(tmp_path):
    text = extract_paper("W1", make_pdf([prose("alpha")]), PARAMS)
    path = paper_file(tmp_path, "W1")
    save_paper_text(path, text)
    assert path == tmp_path / "extraction" / "W1.json"
    assert load_paper_text(path) == text
    assert not list(path.parent.glob("*.part"))   # pas de fichier temporaire oublié


def test_load_returns_none_for_missing_or_broken_files(tmp_path):
    assert load_paper_text(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{pas du json", encoding="utf-8")
    assert load_paper_text(bad) is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"paper_id": "x"}), encoding="utf-8")
    assert load_paper_text(invalid) is None