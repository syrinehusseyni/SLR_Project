"""Corrections de la v3, issues du run demo : mauvais chunk_id, décimaux « 0,008 », métriques traduites, réponse vide, JSON invalide."""
import json

from slrresearch.core.llm import LLMClient
from slrresearch.core.schemas.extraction import Chunk
from slrresearch.extraction.study_card import (
    FIELDS, Answer, build_repair_prompt, fill_field, load_prompt, numbers_in, unsupported_terms, verify_answer,
)
from tests.fakes import FakeLLM

SPEC = FIELDS[0]
FIRST = Chunk(paper_id="P", chunk_id="P_p2_c1", page=2, text="Intro text about something else entirely, nothing about the method here at all.")
SECOND = Chunk(paper_id="P", chunk_id="P_p7_c3", page=7, text="We conducted a comparative experiment to evaluate prompting strategies on three datasets.")
QUOTE = "We conducted a comparative experiment to evaluate prompting strategies"


def client(*responses):
    return LLMClient(FakeLLM(list(responses)), model_name="m", role="extraction", sleep=lambda s: None)


def answer_json(chunk_id, quote=QUOTE, value="réponse"):
    return json.dumps({"status": "found", "chunk_id": chunk_id, "quote": quote, "value": value})


# -- le LLM se trompe de chunk_id : la citation fait foi -------------------------------------------------------
def test_quote_found_in_another_shown_chunk_corrects_the_chunk_id_and_the_page():
    f = verify_answer(Answer("found", "v", "P_p2_c1", QUOTE), SPEC, [FIRST, SECOND])
    assert (f.status, f.chunk_id, f.page) == ("verified", "P_p7_c3", 7)
    assert "corrigé" in f.note and "P_p2_c1" in f.note and "P_p7_c3" in f.note


def test_unknown_chunk_id_with_a_quote_present_in_a_shown_chunk_is_corrected_too():
    f = verify_answer(Answer("found", "v", "Z_p9_c9", QUOTE), SPEC, [FIRST, SECOND])
    assert (f.status, f.chunk_id, f.page) == ("verified", "P_p7_c3", 7)


def test_correct_chunk_id_needs_no_correction():
    f = verify_answer(Answer("found", "v", "P_p7_c3", QUOTE), SPEC, [FIRST, SECOND])
    assert f.status == "verified" and f.note == ""


def test_quote_absent_from_every_shown_chunk_is_still_rejected():
    absent = "this sentence is not in any of the shown chunks, whatever the identifier"
    known = verify_answer(Answer("found", "v", "P_p2_c1", absent), SPEC, [FIRST, SECOND])
    assert known.status == "rejected" and "absente du chunk cité (et des autres extraits)" in known.note
    unknown = verify_answer(Answer("found", "v", "Z_p9_c9", absent), SPEC, [FIRST, SECOND])
    assert unknown.status == "rejected" and "chunk_id absent" in unknown.note


def test_a_quote_from_a_chunk_that_was_not_shown_is_never_corrected_into_it():
    f = verify_answer(Answer("found", "v", "P_p9_c1", "A passage that the model never saw in its context window"), SPEC, [FIRST, SECOND])
    assert f.status == "rejected" and f.page is None and "absente" in f.note


def test_wrong_chunk_id_costs_no_repair_call():
    llm = client(answer_json("P_p2_c1"))
    f = fill_field(llm, SPEC, [FIRST, SECOND], load_prompt(), top_k=3)
    assert f.status == "verified" and f.chunk_id == "P_p7_c3" and llm.calls == 1


# -- nombres : « 0,008 » est un décimal ------------------------------------------------------------------------
def test_leading_zero_decimals_are_not_thousands():
    assert numbers_in("0,008, 0,011, 0,039 et 0,0008") == {"0.008", "0.011", "0.039", "0.0008"}
    assert numbers_in("0,050") == {"0.050"}


def test_real_thousands_are_still_recognised():
    assert numbers_in("10,320 et 1,001") == {"10320", "10", "320", "1001", "1", "001"}
    assert numbers_in("168 842") == {"168842", "168", "842"}


def test_decimals_in_value_and_quote_match_whatever_the_decimal_mark():
    quote = "The p-values were 0.008, 0.011 and 0.039, all below 0.050 in the main analysis."
    chunk = Chunk(paper_id="P", chunk_id="P_p3_c1", page=3, text=quote)
    f = verify_answer(Answer("found", "p-values de 0,008, 0,011 et 0,039, toutes sous 0,050", "P_p3_c1", quote), SPEC, [chunk])
    assert f.status == "verified", f.note


# -- termes : métriques traduites par notre prompt, séparateurs, accents ----------------------------------------
def test_metric_names_translated_as_the_prompt_asks_are_never_suspect():
    assert unsupported_terms("Rappel=1.0, Précision=0.05, Exactitude 84 %, Sensibilité et Spécificité", "recall = 1.0 precision = 0.05") == []


def test_equals_and_slash_separate_words():
    assert unsupported_terms("Valeurs F1=0.9 et GPT-4/Claude", "F1=0.9 with GPT-4") == ["Claude"]


def test_an_invented_model_name_is_still_flagged():
    assert unsupported_terms("Compare Alpaca et Guanaco", "LlaMa outperforms GPT-3 across benchmarks") == ["Alpaca", "Guanaco"]


def test_accents_and_case_are_ignored_when_matching_a_term():
    assert unsupported_terms("Utilise le Réseau Siamois", "a siamois network and a reseau were used") == []


# -- réponse vide et JSON invalide -----------------------------------------------------------------------------
def test_empty_answer_twice_is_an_llm_error_retried_on_resume_not_invalid_format():
    for empty in ("", "   \n"):
        llm = client(empty, empty)
        f = fill_field(llm, SPEC, [FIRST, SECOND], load_prompt(), top_k=3)
        assert f.status == "llm_error" and "vide" in f.note and llm.calls == 2


def test_empty_answer_then_good_answer_is_verified():
    llm = client("", answer_json("P_p7_c3"))
    assert fill_field(llm, SPEC, [FIRST, SECOND], load_prompt(), top_k=3).status == "verified"


def test_garbage_twice_is_still_invalid_format():
    llm = client("pas du json", "toujours pas")
    assert fill_field(llm, SPEC, [FIRST, SECOND], load_prompt(), top_k=3).status == "invalid_format"


def test_repair_after_invalid_json_explains_how_to_escape_quotes():
    broken = '{"status": "found", "chunk_id": "P_p7_c3", "quote": "he said "yes" to it", "value": "v"}'
    llm = client(broken, answer_json("P_p7_c3"))
    assert fill_field(llm, SPEC, [FIRST, SECOND], load_prompt(), top_k=3).status == "verified"
    assert "guillemet double" in llm.model.prompts[1] and "JSON invalide" in llm.model.prompts[1]


def test_repair_hint_only_for_json_errors():
    assert "guillemet double" in build_repair_prompt("P", "{", "JSON invalide : Expecting ','")
    assert "guillemet double" not in build_repair_prompt("P", "{}", "citation trop courte")