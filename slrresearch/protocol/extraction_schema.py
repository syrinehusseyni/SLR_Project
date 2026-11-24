"""Facettes : informations requises pour répondre à chaque question de recherche."""
from slrresearch.core.schemas.protocol_plan import FacetSpec
from slrresearch.protocol.common import clean_lines, generate_section, questions_block

MIN_PER_QUESTION = 1
MAX_PER_QUESTION = 6
_YES = {"yes", "oui", "true", "1"}
_NO = {"no", "non", "false", "0"}


def build_facets_prompt(topic: str, questions) -> str:
    return (
        "Tu prépares le formulaire d'extraction d'une revue systématique de la littérature.\n"
        f"Sujet : {topic}\n"
        "Questions de recherche :\n"
        f"{questions_block(questions)}\n\n"
        f"Pour CHAQUE question, liste {MIN_PER_QUESTION} à {MAX_PER_QUESTION} FACETTES : les informations qu'il faut "
        "trouver dans un article pour pouvoir répondre (par exemple : type d'étude, jeu de données, modèle utilisé, "
        "métrique, valeur obtenue, limites).\n"
        "Format EXACT, une ligne par facette : RQn | étiquette courte | description | bloquante\n"
        "où « bloquante » vaut oui (sans cette information on ne peut pas répondre) ou non (utile mais facultative).\n"
        "Exemple : RQ1 | Métrique | Mesure de performance rapportée (rappel, précision, F1) | oui\n"
        "Aucune introduction, aucun commentaire."
    )


def build_facets_repair_prompt(text: str, error: str) -> str:
    return (
        "Ta réponse précédente ne respecte pas le format demandé.\n"
        f"Problème détecté : {error}\n"
        "Réponse précédente :\n"
        f"{text}\n\n"
        "Corrige-la. Une ligne par facette : RQn | étiquette | description | oui ou non. "
        "Chaque question doit avoir au moins une facette. Réponds uniquement avec les lignes."
    )


def parse_facets(text: str, question_ids: list[str]) -> list[FacetSpec]:
    """Lève ValueError. Les identifiants F1..Fn sont attribués ici."""
    lines = clean_lines(text)
    if not lines:
        raise ValueError("réponse vide")
    facets: list[FacetSpec] = []
    for number, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"ligne {number} non conforme (attendu : RQn | étiquette | description | oui/non) : {line!r}")
        rq_id = parts[0].upper()
        if rq_id not in question_ids:
            raise ValueError(f"ligne {number} : question inconnue {parts[0]!r} (questions : {', '.join(question_ids)})")
        flag = parts[3].lower().rstrip(".")
        if flag in _YES:
            blocking = True
        elif flag in _NO:
            blocking = False
        else:
            raise ValueError(f"ligne {number} : « bloquante » doit valoir oui ou non, reçu {parts[3]!r}")
        try:
            facets.append(FacetSpec(id=f"F{number}", rq_id=rq_id, label=parts[1], description=parts[2], blocking=blocking))
        except ValueError as exc:
            raise ValueError(f"ligne {number} : {exc}") from None
    for rq_id in question_ids:
        count = sum(1 for f in facets if f.rq_id == rq_id)
        if count < MIN_PER_QUESTION:
            raise ValueError(f"la question {rq_id} n'a aucune facette")
        if count > MAX_PER_QUESTION:
            raise ValueError(f"la question {rq_id} a {count} facettes, maximum {MAX_PER_QUESTION}")
    return facets


def generate_facets(llm, topic: str, questions, trace=None):
    question_ids = [q.identifier for q in questions]
    return generate_section(
        llm,
        build_facets_prompt(topic, questions),
        lambda text: parse_facets(text, question_ids),
        build_facets_repair_prompt,
        trace=trace,
        section="facets",
    )