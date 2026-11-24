"""15 champs `verified` tirés au hasard du run demo (v1), relus à la main : les contrôles de la v2 doivent refuser ceux dont la
valeur ajoute un nom ou un nombre absent de la citation, et ne pas refuser ceux qui sont corrects."""
import pytest

from slrresearch.core.schemas.extraction import Chunk
from slrresearch.extraction.study_card import FIELDS, Answer, verify_answer

FIELD = {f.id: f for f in FIELDS}

# (id, champ, valeur, citation, verdict de la relecture manuelle, contrôle automatique attendu)
CASES = [
    (1, "data", "8 jeux de données, domaine biomédical",
     "While the 8 datasets span different drug classes with varying inclusion rates and sizes, all are within the biomedical domain.", "ok", "verified"),
    (2, "limitations", "Les auteurs signalent que ChatGPT ne fournit pas de références vérifiables et nécessite une expertise méthodologique pour corriger ses erreurs.",
     "A particularly strong limitation of the system is the lack of referencing appropriate and verifiable sources when asked for factual information.", "partiel", "rejected"),
    (3, "design", "étude expérimentale comparative utilisant un protocole d’inférence multi-pass avec températures de sampling T ∈{0.1, 0.4, 0.8}",
     "We performed three independent inference passes using sampling temperatures T ∈{0.1, 0.4, 0.8}.", "partiel", "verified"),  # « comparative » : non détecté
    (4, "models", "Le modèle évalué est GPT\u20114.0 (ChatGPT)",
     "On March 15, 2023, GPT-4.0 was released for testing among OpenAI’s paid subscribers [18].", "partiel", "rejected"),
    (5, "models", "Les modèles utilisés sont GPT\u20116 Astra (OpenAI) et Claude Fable 5.1 (Anthropic), ainsi que le modèle Jev.",
     "We used GPT-6 Astra (OpenAI) and Claude Fable 5.1 (Anthropic) to assist with prompt development, code drafting, analysis, and manuscript preparation.", "faux", "rejected"),
    (6, "findings", "DFS a obtenu les meilleurs scores F1 sur les dix revues systématiques, surpassant Zero-Shot, Chain-of-Thought et Few-Shot.",
     "DFS achieved higher F1 scores in the title and abstract screening task across all ten systematic reviews compared to Zero-Shot, Chain-of-Thought, and Few-Shot methods.", "ok", "verified"),
    (7, "findings", "GPTscreenR a une sensibilité de 71 %, une spécificité de 89 % et une précision globale de 84 %.",
     "GPTscreenR performed similarly to an alternative zero-shot technique, with a sensitivity of 71%, specificity of 89%, and overall accuracy of 84%.", "partiel", "verified"),  # accuracy/précision : par le prompt
    (8, "limitations", "Risques de hallucinations pouvant inclure références fabriquées, citations erronées ou interprétations incorrectes, menaçant l'intégrité académique.",
     "Hallucinated outputs can include fabricated references, misquoted findings, or incorrect interpretations, each of which poses a direct threat to the integrity of academic work (Fui-Hoon Nah et al. 2023).", "ok", "verified"),
    (9, "data", "Le jeu de données Instruct Cochrane contient 168\u202f842 échantillons.",
     "Total 168,842 1,001 1,711 108 780", "partiel", "rejected"),
    (10, "limitations", "Temps plus long et coût plus élevé, risque de surapprentissage.",
     "This results in both a longer time to screen a single source (typically 20–30 seconds) and a higher dollar cost, as the OpenAI API currently charges on a per-token basis.", "partiel", "verified"),  # non détecté : par le prompt et le juge
    (11, "data", "CLEF 2019 dataset, 10,320 titles/abstracts retrieved, 9,515 curated, 1,052 included, covering diagnosis, intervention, prognosis.",
     "the total number of titles and abstracts generated from the search step was 10,320. After data curation, the actual number of titles and abstracts included was 9,515, of which 1,052 were identified as ones should be included in their corresponding systematic reviews during the title and abstract screening stage.", "partiel", "rejected"),
    (12, "models", "Les modèles évalués sont LlaMa, Alpaca, Guanaco, Falcon, LlaMa2 et BioBERT.",
     "LlaMa: The LlaMa series offers an open-sourced suite of decoder models with parameter sizes ranging from 7B to 65B. Exceptional in its zero-shot capabilities, LlaMa outperforms GPT-3 across multiple NLP benchmarks.", "partiel", "rejected"),
    (13, "data", "1381 records from a systematic review of bright light therapy for bipolar disorder and 3146 records from a systematic review of adjunctive pharmacotherapy for bipolar depression, totaling 4527 records.",
     "We analyzed two datasets previously used to evaluate GPT-3.5 and GPT-410: 1381 records from a systematic review of bright light therapy for bipolar disorder (hereafter, light therapy)24 and 3146 from a systematic review of adjunctive pharmacotherapy for bipolar depression.25 Eligibility criteria", "ok", "rejected"),  # somme calculée : refusée, la réparation demandera de la retirer
    (14, "limitations", "Bio-SIEVE présente des problèmes de cohérence sévères entre les sujets.",
     "Bio-SIEVE also displayed severe consistency issues across review topics.", "partiel", "verified"),  # limite d'un autre système : par scope=other
    (15, "design", "revue systématique",
     "whether a paper is relevant for a narrow systematic review that collects primary, peer-reviewed empirical studies that build, evaluate, or compare machine-learning models for predicting software defects/bugs/faults/failures.", "faux", "verified"),  # annexe de prompt : par scope=other
]


@pytest.mark.parametrize("num,field,value,quote,judgement,expected", CASES, ids=[f"cas{c[0]}-{c[4]}" for c in CASES])
def test_real_case(num, field, value, quote, judgement, expected):
    chunk = Chunk(paper_id="P", chunk_id="P_p1_c1", page=1, text=quote)
    result = verify_answer(Answer("found", value, "P_p1_c1", quote), FIELD[field], [chunk])
    assert result.status == expected, result.note


def test_no_correct_answer_is_refused_except_the_derived_sum():
    refused_ok = [c[0] for c in CASES if c[4] == "ok" and not verify_ok(c)]
    assert refused_ok == [13]


def test_wrong_answers_caught_by_the_program():
    caught = [c[0] for c in CASES if c[4] in ("partiel", "faux") and not verify_ok(c)]
    assert caught == [2, 4, 5, 9, 11, 12]          # 6 des 11 réponses douteuses ; les 5 autres relèvent du prompt, du scope et du juge (étape 6)


def verify_ok(case):
    _, field, value, quote, *_ = case
    chunk = Chunk(paper_id="P", chunk_id="P_p1_c1", page=1, text=quote)
    return verify_answer(Answer("found", value, "P_p1_c1", quote), FIELD[field], [chunk]).status == "verified"