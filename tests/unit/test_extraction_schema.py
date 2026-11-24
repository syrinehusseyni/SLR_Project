import pytest
from pydantic import ValidationError

from slrresearch.core.schemas.extraction import (
    Chunk, ExtractedPaperEntry, ExtractionLimits, ExtractionOutcome, ExtractionParams, ExtractionSummary, PaperText,
)

PARAMS = ExtractionParams(chunk_size=100, chunk_overlap=10, max_pages=None, min_chars_per_page=50, cleaning="1")


def chunk(cid="P_p1_c1", page=1, paper="P"):
    return Chunk(paper_id=paper, chunk_id=cid, page=page, text="x")


def text(**kw):
    base = dict(paper_id="P", source_sha256="h", params=PARAMS, total_pages=2, pages_read=2, page_chars=[5, 0], chunks=[chunk()])
    return PaperText(**{**base, **kw})


def test_chunk_id_must_match_paper_and_page():
    chunk()
    for cid, page, paper in [("P_p2_c1", 1, "P"), ("Q_p1_c1", 1, "P"), ("bad", 1, "P"), ("P_p1", 1, "P")]:
        with pytest.raises(ValidationError):
            chunk(cid, page, paper)


def test_chunk_id_with_underscores_in_paper_id():
    Chunk(paper_id="doi_10_x", chunk_id="doi_10_x_p3_c2", page=3, text="t")


def test_params_overlap_below_size():
    with pytest.raises(ValidationError):
        ExtractionParams(chunk_size=10, chunk_overlap=10, min_chars_per_page=0, cleaning="1")


def test_papertext_valid_and_truncated_flag():
    assert text().truncated is False
    assert text(total_pages=9).truncated is True


@pytest.mark.parametrize("bad", [
    dict(page_chars=[1]),                                  # une valeur par page lue
    dict(pages_read=3, total_pages=2, page_chars=[1, 2, 3]),
    dict(chunks=[chunk(), chunk()]),                       # chunk_id en double
    dict(chunks=[chunk("P_p3_c1", 3)]),                    # page au-delà des pages lues
    dict(chunks=[chunk("Q_p1_c1", 1, "Q")]),               # autre papier
    dict(chunks=[]),
])
def test_papertext_rejects_inconsistencies(bad):
    with pytest.raises(ValidationError):
        text(**bad)


def test_entry_rules():
    ExtractedPaperEntry(paper_id="P", status="extracted", file="extraction/P.json", source_sha256="h", chunks=3)
    ExtractedPaperEntry(paper_id="P", status="failed", reason="illisible")
    with pytest.raises(ValidationError):
        ExtractedPaperEntry(paper_id="P", status="extracted")
    with pytest.raises(ValidationError):
        ExtractedPaperEntry(paper_id="P", status="failed")


def _outcome(**kw):
    ok = ExtractedPaperEntry(paper_id="A", status="extracted", file="f", source_sha256="h", chunks=4)
    ko = ExtractedPaperEntry(paper_id="B", status="failed", reason="r")
    base = dict(limits=ExtractionLimits(**PARAMS.model_dump()), papers=[ok, ko],
                summary=ExtractionSummary(total=2, extracted=1, failed=1, chunks=4, skipped_unavailable=0))
    return ExtractionOutcome(**{**base, **kw})


def test_outcome_consistency():
    _outcome()
    with pytest.raises(ValidationError):
        ExtractionSummary(total=3, extracted=1, failed=1, chunks=0, skipped_unavailable=0)
    with pytest.raises(ValidationError):
        _outcome(summary=ExtractionSummary(total=2, extracted=1, failed=1, chunks=99, skipped_unavailable=0))
    with pytest.raises(ValidationError):
        _outcome(summary=ExtractionSummary(total=2, extracted=2, failed=0, chunks=4, skipped_unavailable=0))