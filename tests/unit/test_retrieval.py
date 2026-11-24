from slrresearch.core.schemas.extraction import Chunk
from slrresearch.extraction.retrieval import lexical_retrieve, tokenize


def chunks(*texts):
    return [Chunk(paper_id="P", chunk_id=f"P_p1_c{i}", page=1, text=t) for i, t in enumerate(texts, 1)]


def test_tokenize_lowercases_and_drops_single_chars():
    assert tokenize("The GPT-4 model: a 0.97 recall!") == ["the", "gpt", "model", "97", "recall"]
    assert tokenize("a b cd") == ["cd"]


def test_most_relevant_chunk_comes_first():
    cs = chunks("weather is sunny today", "recall and precision were measured on the test set", "nothing here at all")
    found = lexical_retrieve(cs, "recall precision test", k=3)
    assert [c.chunk_id for c, _ in found][0] == "P_p1_c2"


def test_zero_score_chunks_are_never_returned():
    cs = chunks("alpha beta", "gamma delta")
    assert [c.chunk_id for c, _ in lexical_retrieve(cs, "alpha", k=5)] == ["P_p1_c1"]
    assert lexical_retrieve(cs, "zzz yyy", k=5) == []


def test_k_limits_and_scores_decrease():
    cs = chunks(*[f"recall word{i} " + "recall " * i for i in range(6)])
    found = lexical_retrieve(cs, "recall", k=3)
    assert len(found) == 3
    scores = [s for _, s in found]
    assert scores == sorted(scores, reverse=True)


def test_ties_keep_document_order():
    cs = chunks("recall here", "recall here", "recall here")
    assert [c.chunk_id for c, _ in lexical_retrieve(cs, "recall", k=3)] == ["P_p1_c1", "P_p1_c2", "P_p1_c3"]


def test_empty_inputs_and_case_insensitive():
    assert lexical_retrieve([], "recall") == []
    assert lexical_retrieve(chunks("recall"), "recall", k=0) == []
    assert lexical_retrieve(chunks("RECALL Precision"), "recall precision", k=1)


def test_query_language_matters():
    cs = chunks("the model reached high recall on the dataset")
    assert lexical_retrieve(cs, "rappel modèle jeu de données", k=3) == []   # requête française : rien dans un texte anglais