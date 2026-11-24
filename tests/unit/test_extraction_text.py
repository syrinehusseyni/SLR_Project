import pytest

from slrresearch.extraction.text import chunk_pages, clean_page_text, find_repeated_lines, validate_chunking


# -- nettoyage -------------------------------------------------------------------------------------------------
def test_clean_joins_hyphenated_line_ends():
    assert clean_page_text("a classi-\nfication task and Specif-  \nically this") == "a classification task and Specifically this"


def test_clean_keeps_hyphen_when_next_line_does_not_start_lowercase():
    # « state-of-the-\nArt » et « COVID-\n19 » : on ne devine pas, on laisse tel quel
    assert "-\nArt" in clean_page_text("state-of-the-\nArt")
    assert "-\n19" in clean_page_text("COVID-\n19")


def test_clean_keeps_real_hyphens_inside_a_line():
    assert clean_page_text("zero-shot and few-shot") == "zero-shot and few-shot"


def test_clean_ligatures_control_chars_and_spaces():
    assert clean_page_text("\ufb01nd \ufb02ow o\ufb00 e\ufb03cient\x00 a\xa0b\xadc   d") == "find flow off efficient a bc d"


def test_clean_collapses_blank_lines_and_strips_lines():
    assert clean_page_text("  one  \n\n\n\n  two  ") == "one\n\ntwo"


def test_clean_is_idempotent():
    once = clean_page_text("classi-\nfication \ufb01x\n\n\n\nend")
    assert clean_page_text(once) == once


# -- en-têtes / pieds de page ----------------------------------------------------------------------------------
def test_repeated_lines_detect_running_header_and_page_numbers():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = [f"Journal of Tests 2024\nbody about {w} unique text here\nPage {i}" for i, w in enumerate(words, 1)]
    found = find_repeated_lines(pages)
    assert "Journal of Tests #" in found and "Page #" in found
    assert not any("unique" in f for f in found)


def test_repeated_lines_need_enough_pages():
    assert find_repeated_lines(["Header line\nx", "Header line\ny"]) == []


def test_repeated_lines_ignore_very_short_lines():
    assert find_repeated_lines(["ab\nz"] * 6) == []


# -- chunks ----------------------------------------------------------------------------------------------------
def test_validate_chunking():
    validate_chunking(10, 0)
    for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
        with pytest.raises(ValueError):
            validate_chunking(size, overlap)


def test_chunk_ids_pages_and_provenance():
    chunks = chunk_pages("W1", ["alpha beta gamma delta", "", "epsilon zeta"], size=12, overlap=3)
    assert [c.chunk_id for c in chunks][0] == "W1_p1_c1"
    assert {c.page for c in chunks} == {1, 3}          # la page 2 est vide : aucun chunk
    assert all(c.chunk_id == f"W1_p{c.page}_c{n}" for page in (1, 3)
               for n, c in enumerate([c for c in chunks if c.page == page], 1))
    assert all(c.paper_id == "W1" for c in chunks)


def test_chunks_never_exceed_size_and_never_cut_words():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_pages("P", [text], size=60, overlap=15)
    words = set(text.split())
    for chunk in chunks:
        assert len(chunk.text) <= 60
        assert set(chunk.text.split()) <= words  # chaque mot est entier


def test_chunks_cover_the_whole_page_with_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_pages("P", [text], size=80, overlap=20)
    assert {w for c in chunks for w in c.text.split()} == set(text.split())  # rien n'est perdu
    for before, after in zip(chunks, chunks[1:]):
        assert set(before.text.split()) & set(after.text.split())            # les voisins se chevauchent


def test_a_quote_inside_a_chunk_is_found_verbatim_with_its_page():
    page2 = "Intro words here. The best model reached recall 0.97 on the test set. Closing words follow."
    chunks = chunk_pages("P", ["page one text", page2], size=60, overlap=20)
    quote = "recall 0.97 on the test"
    hits = [c for c in chunks if quote in c.text]
    assert hits and all(c.page == 2 for c in hits)


def test_overlap_zero_and_single_short_page():
    assert [c.text for c in chunk_pages("P", ["short"], size=100, overlap=0)] == ["short"]
    parts = chunk_pages("P", ["aaa bbb ccc ddd"], size=8, overlap=0)
    assert " ".join(c.text for c in parts) == "aaa bbb ccc ddd"


def test_very_long_token_is_hard_cut_and_terminates():
    chunks = chunk_pages("P", ["x" * 100], size=30, overlap=5)
    assert all(len(c.text) <= 30 for c in chunks) and sum(len(c.text) for c in chunks) >= 100


def test_chunking_is_deterministic():
    pages = ["some text " * 50, "other text " * 40]
    assert chunk_pages("P", pages, 100, 20) == chunk_pages("P", pages, 100, 20)