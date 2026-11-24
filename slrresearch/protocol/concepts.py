"""Concepts de recherche : groupes de synonymes proposés par le LLM, contrôlés par le programme."""
from slrresearch.core.schemas.protocol_plan import SearchConcept
from slrresearch.protocol.common import clean_lines, generate_section, questions_block

MIN_CONCEPTS = 2
MAX_CONCEPTS = 4


def build_concepts_prompt(topic: str, questions) -> str:
    return (
        "Tu prépares la stratégie de recherche d'une revue systématique de la littérature.\n"
        f"Sujet : {topic}\n"
        "Questions de recherche :\n"
        f"{questions_block(questions)}\n\n"
        f"Découpe le sujet en {MIN_CONCEPTS} à {MAX_CONCEPTS} CONCEPTS de recherche. "
        "Un concept est un groupe de termes équivalents (synonymes, variantes d'écriture, sigles). "
        "Plus tard, les termes d'un même concept seront reliés par OR et les concepts entre eux par AND.\n"
        "Règles :\n"
        "- termes en ANGLAIS (la littérature scientifique est surtout en anglais) ;\n"
        "- 3 à 6 termes par concept, le plus important en premier ;\n"
        "- ni guillemets, ni parenthèses, ni mots AND / OR / NOT dans les termes ;\n"
        "- une ligne par concept, format EXACT : étiquette | terme 1; terme 2; terme 3\n"
        "- TEXTE BRUT uniquement : pas de markdown, pas de gras (**), pas de titres, pas de puces, "
        "pas de numérotation, pas de « Concept 1 », aucune introduction, aucun commentaire.\n\n"
        "Exemple pour le sujet « deep learning pour détecter les maladies des plantes » :\n"
        "task | plant disease detection; crop disease diagnosis; leaf disease classification\n"
        "method | deep learning; convolutional neural network; CNN; computer vision\n\n"
        f"Réponds maintenant avec {MIN_CONCEPTS} à {MAX_CONCEPTS} lignes de ce format, et rien d'autre."
    )


def build_concepts_repair_prompt(text: str, error: str) -> str:
    return (
        "Ta réponse précédente ne respecte pas le format demandé.\n"
        f"Problème détecté : {error}\n"
        "Réponse précédente :\n"
        f"{text}\n\n"
        "Corrige-la. Une ligne par concept, format EXACT : étiquette | terme 1; terme 2; terme 3\n"
        "Exemple de réponse correcte (texte brut, sans gras, sans titre) :\n"
        "task | plant disease detection; crop disease diagnosis\n"
        "method | deep learning; convolutional neural network; CNN\n"
        f"Entre {MIN_CONCEPTS} et {MAX_CONCEPTS} lignes, termes en anglais, sans guillemets ni parenthèses. "
        "Réponds uniquement avec les lignes."
    )


def parse_concepts(text: str) -> list[SearchConcept]:
    """Lève ValueError si le texte ne respecte pas le format. Les identifiants C1..Cn sont attribués ici."""
    lines = clean_lines(text)
    if not lines:
        raise ValueError("réponse vide")
    concepts: list[SearchConcept] = []
    for number, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"ligne {number} non conforme (attendu : étiquette | terme; terme) : {line!r}")
        terms = [t.strip() for t in parts[1].split(";") if t.strip()]
        try:
            concepts.append(SearchConcept(id=f"C{number}", label=parts[0], terms=terms))
        except ValueError as exc:
            raise ValueError(f"ligne {number} : {exc}") from None
    if not MIN_CONCEPTS <= len(concepts) <= MAX_CONCEPTS:
        raise ValueError(f"{len(concepts)} concept(s) reçu(s), il en faut entre {MIN_CONCEPTS} et {MAX_CONCEPTS}")
    labels = [c.label.lower() for c in concepts]
    if len(labels) != len(set(labels)):
        raise ValueError("deux concepts ont la même étiquette")
    return concepts


def generate_concepts(llm, topic: str, questions, trace=None):
    return generate_section(
        llm,
        build_concepts_prompt(topic, questions),
        parse_concepts,
        build_concepts_repair_prompt,
        trace=trace,
        section="concepts",
    )