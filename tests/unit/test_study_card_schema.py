import pytest
from pydantic import ValidationError

from slrresearch.core.schemas.study_card import (
    CardEntry, CardField, StudyCard, StudyCardsLimits, StudyCardsOutcome, StudyCardsSummary,
)

LIMITS = StudyCardsLimits(model="m", prompt="study_card.txt", top_k=3, pause=0, fields=["design"])


def field(fid="design"):
    return CardField(field_id=fid, label="L", status="not_reported")


def test_card_needs_unique_fields():
    StudyCard(paper_id="P", fingerprint="f", model="m", fields=[field("a"), field("b")])
    with pytest.raises(ValidationError):
        StudyCard(paper_id="P", fingerprint="f", model="m", fields=[field("a"), field("a")])
    with pytest.raises(ValidationError):
        StudyCard(paper_id="P", fingerprint="f", model="m", fields=[])


def test_entry_rules():
    CardEntry(paper_id="P", status="complete", file="study_cards/P.json")
    CardEntry(paper_id="P", status="failed", reason="illisible")
    for bad in (dict(status="failed"), dict(status="complete"), dict(status="incomplete")):
        with pytest.raises(ValidationError):
            CardEntry(paper_id="P", **bad)


def _outcome(**kw):
    ok = CardEntry(paper_id="A", status="complete", file="f")
    ko = CardEntry(paper_id="B", status="failed", reason="r")
    base = dict(limits=LIMITS, papers=[ok, ko],
                summary=StudyCardsSummary(total=2, complete=1, incomplete=0, failed=1, skipped_not_extracted=0, llm_calls=5, tokens=0))
    return StudyCardsOutcome(**{**base, **kw})


def test_outcome_consistency():
    _outcome()
    with pytest.raises(ValidationError):
        StudyCardsSummary(total=3, complete=1, incomplete=0, failed=1, skipped_not_extracted=0, llm_calls=0, tokens=0)
    with pytest.raises(ValidationError):
        _outcome(summary=StudyCardsSummary(total=2, complete=2, incomplete=0, failed=0, skipped_not_extracted=0, llm_calls=0, tokens=0))
    with pytest.raises(ValidationError):
        _outcome(papers=[CardEntry(paper_id="A", status="complete", file="f")] * 2)