import pytest

from slrresearch.core.errors import ReviewError
from slrresearch.hitl.decisions import DecisionLog
from slrresearch.hitl.queue import ReviewQueue


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(tmp_path / "hitl.jsonl", DecisionLog(tmp_path / "decisions.jsonl"))


def open_g2(queue, target="W1"):
    return queue.open_item("G2", "paper", target, "desaccord entre regles et LLM")


def test_open_creates_open_item_with_stable_id(queue):
    item = open_g2(queue)
    assert item.id == "G2:paper:W1"
    assert item.status == "open"
    assert queue.has_open("G2")


def test_opening_same_item_twice_does_not_duplicate(queue):
    open_g2(queue)
    open_g2(queue)
    assert len(queue.list_items()) == 1


def test_close_with_justification_works(queue):
    item = open_g2(queue)
    closed = queue.close_item(item.id, "include", "rahma", "le resume decrit bien un LLM pour le screening")
    assert closed.status == "closed"
    assert closed.decision == "include"
    assert closed.decided_by == "rahma"
    assert closed.decided_at
    assert not queue.has_open("G2")


@pytest.mark.parametrize("justification", ["", "   ", None])
def test_close_without_justification_is_refused_and_item_stays_open(queue, justification):
    item = open_g2(queue)
    with pytest.raises(ReviewError, match="justification"):
        queue.close_item(item.id, "include", "rahma", justification)
    assert queue.get(item.id).status == "open"
    assert queue.decisions.read() == []  # rien n'est journalise


def test_close_without_author_or_decision_is_refused(queue):
    item = open_g2(queue)
    with pytest.raises(ReviewError, match="decided_by"):
        queue.close_item(item.id, "include", "", "raison")
    with pytest.raises(ReviewError, match="decision"):
        queue.close_item(item.id, "", "rahma", "raison")


def test_cannot_close_twice(queue):
    item = open_g2(queue)
    queue.close_item(item.id, "include", "rahma", "raison")
    with pytest.raises(ReviewError, match="deja ferme"):
        queue.close_item(item.id, "exclude", "rahma", "autre raison")


def test_close_unknown_item_refused(queue):
    with pytest.raises(ReviewError, match="inconnu"):
        queue.close_item("G2:paper:nope", "include", "rahma", "raison")


def test_invalid_gate_refused(queue):
    with pytest.raises(ReviewError, match="invalide"):
        queue.open_item("G9", "paper", "W1", "raison")


def test_reopening_a_closed_item_returns_it_closed_so_human_is_not_asked_again(queue):
    item = open_g2(queue)
    queue.close_item(item.id, "include", "rahma", "raison")
    again = open_g2(queue)  # l'etape est relancee
    assert again.status == "closed"
    assert not queue.has_open()


def test_state_survives_reopening_the_file(tmp_path):
    path = tmp_path / "hitl.jsonl"
    q1 = ReviewQueue(path)
    item = q1.open_item("G3", "document", "W9", "PDF scanne")
    q1.close_item(item.id, "manual_upload", "rahma", "PDF fourni a la main")
    q2 = ReviewQueue(path)  # nouvelle instance = reprise apres arret
    assert q2.get(item.id).status == "closed"


def test_file_is_append_only(tmp_path):
    path = tmp_path / "hitl.jsonl"
    q = ReviewQueue(path)
    item = q.open_item("G1", "protocol", "p1", "valider le protocole")
    q.close_item(item.id, "approve", "rahma", "criteres corrects")
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2  # ouverture + fermeture


def test_filter_by_gate_and_status(queue):
    queue.open_item("G2", "paper", "W1", "r")
    queue.open_item("G3", "document", "W1", "r")
    assert [i.gate for i in queue.open_items("G3")] == ["G3"]
    assert queue.has_open() and queue.has_open("G2")
    queue.close_item("G2:paper:W1", "include", "rahma", "raison")
    assert not queue.has_open("G2")
    assert queue.has_open("G3")


def test_decision_journal_records_who_when_what_why(queue):
    item = open_g2(queue)
    queue.close_item(item.id, "exclude", "rahma", "hors sujet : parle d'imagerie medicale")
    (decision,) = queue.decisions.read()
    assert decision.item_id == item.id
    assert decision.decided_by == "rahma"
    assert decision.decision == "exclude"
    assert "imagerie" in decision.justification
    assert decision.decided_at