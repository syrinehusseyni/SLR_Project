import pytest

from slrresearch.download.validate import (
    extract_pages, looks_like_pdf, title_coverage, title_words, validate_pdf,
)
from tests.pdf_helpers import BODY, TITLE, good_pdf, make_pdf

BACKENDS = ["pymupdf", "pypdf"]


def available(backend):
    pytest.importorskip("pymupdf" if backend == "pymupdf" else "pypdf")


def test_title_words_drop_short_and_common_words():
    assert title_words("The Use of GPT-4 for the Screening of Reviews") == ["use", "gpt", "screening", "reviews"]


def test_title_coverage_whole_words_and_accents():
    assert title_coverage("Évaluation des revues", "evaluation revues systematiques") == 1.0
    assert title_coverage("alpha beta gamma delta", "alpha beta") == 0.5
    assert title_coverage("screen", "screening") == 0.0  # mots entiers, pas de sous-chaîne
    assert title_coverage("a of", "n'importe quoi") == 1.0  # aucun mot utile : rien à vérifier


def test_looks_like_pdf():
    assert looks_like_pdf(b"%PDF-1.7 ...")
    assert looks_like_pdf(b"\n  %PDF-1.4")
    assert not looks_like_pdf(b"<!DOCTYPE html><html>")
    assert not looks_like_pdf(b"")


@pytest.mark.parametrize("backend", BACKENDS)
def test_extract_pages_reads_text_and_page_count(backend):
    available(backend)
    pages, total = extract_pages(make_pdf(["bonjour alpha", "page deux", "trois"]), max_pages=2, backend=backend)
    assert total == 3 and len(pages) == 2
    assert "alpha" in pages[0] and "deux" in pages[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_ok(backend):
    available(backend)
    check = validate_pdf(good_pdf(), TITLE, backend=backend)
    assert check.ok and check.verdict == "ok" and check.pages == 3
    assert check.title_coverage == 1.0 and check.avg_chars > 50


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_wrong_article(backend):
    available(backend)
    other = make_pdf(["Fine tuning of convolutional networks for tumour detection\n" + BODY, BODY])
    check = validate_pdf(other, TITLE, backend=backend)
    assert not check.ok and check.verdict == "wrong_title" and check.title_coverage < 0.7


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_scanned_pdf_has_no_text_layer(backend):
    available(backend)
    check = validate_pdf(make_pdf(["", "", ""]), TITLE, backend=backend)
    assert not check.ok and check.verdict == "no_text_layer" and check.avg_chars < 50


def test_validate_html_page_is_not_pdf():
    check = validate_pdf(b"<html><body>Please log in</body></html>", TITLE)
    assert not check.ok and check.verdict == "not_pdf"


def test_validate_too_large_is_checked_first():
    check = validate_pdf(good_pdf(), TITLE, max_bytes=100)
    assert check.verdict == "too_large"


def test_validate_corrupt_pdf_is_unreadable():
    check = validate_pdf(b"%PDF-1.4\nce n'est pas un vrai pdf", TITLE)
    assert not check.ok and check.verdict == "unreadable"


def test_validate_thresholds_are_parameters():
    other = make_pdf(["Fine tuning of convolutional networks\n" + BODY])
    assert validate_pdf(other, TITLE).verdict == "wrong_title"
    assert validate_pdf(other, TITLE, min_title_coverage=0.0).ok
    assert validate_pdf(good_pdf(), TITLE, min_chars_per_page=10_000).verdict == "no_text_layer"