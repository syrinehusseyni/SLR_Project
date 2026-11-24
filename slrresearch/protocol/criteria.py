"""Critères d'inclusion / d'exclusion proposés par le LLM, validés par le programme."""
from slrresearch.core.schemas.protocol_plan import Criteria, Criterion
from slrresearch.protocol.common import clean_lines, generate_section, questions_block

MIN_PER_KIND = 2
MAX_PER_KIND = 6


def build_criteria_prompt(topic: str, questions) -> str:
    return (
        "Tu prépares le protocole d'une revue systématique de la littérature.\n"
        f"Sujet : {topic}\n"
        "Questions de recherche :\n"
        f"{questions_block(questions)}\n\n"
        f"Propose {MIN_PER_KIND} à {MAX_PER_KIND} critères d'INCLUSION et {MIN_PER_KIND} à {MAX_PER_KIND} critères "
        "d'EXCLUSION, applicables en lisant uniquement le titre et le résumé d'un article.\n"
        "Chaque critère est une phrase courte, vérifiable, en français.\n"
        "Format EXACT, une ligne par critère :\n"
        "INCLUDE | critère d'inclusion\n"
        "EXCLUDE | critère d'exclusion\n"
        "Aucune introduction, aucun commentaire, aucune numérotation."
    )


def build_criteria_repair_prompt(text: str, error: str) -> str:
    return (
        "Ta réponse précédente ne respecte pas le format demandé.\n"
        f"Problème détecté : {error}\n"
        "Réponse précédente :\n"
        f"{text}\n\n"
        "Corrige-la. Une ligne par critère, début de ligne INCLUDE | ou EXCLUDE |, "
        f"entre {MIN_PER_KIND} et {MAX_PER_KIND} lignes de chaque type. Réponds uniquement avec les lignes."
    )


def parse_criteria(text: str, year_from: int | None = None, languages=("en",)) -> Criteria:
    """Lève ValueError. Les identifiants I1.. / E1.. sont attribués ici ; année et langues viennent de la config."""
    lines = clean_lines(text)
    if not lines:
        raise ValueError("réponse vide")
    inclusion: list[Criterion] = []
    exclusion: list[Criterion] = []
    for number, line in enumerate(lines, start=1):
        head, sep, body = line.partition("|")
        kind = head.strip().upper()
        body = body.strip()
        if not sep or kind not in ("INCLUDE", "EXCLUDE") or not body:
            raise ValueError(f"ligne {number} non conforme (attendu : INCLUDE | ... ou EXCLUDE | ...) : {line!r}")
        try:
            if kind == "INCLUDE":
                inclusion.append(Criterion(id=f"I{len(inclusion) + 1}", kind="include", text=body))
            else:
                exclusion.append(Criterion(id=f"E{len(exclusion) + 1}", kind="exclude", text=body))
        except ValueError as exc:
            raise ValueError(f"ligne {number} : {exc}") from None
    for name, items in (("INCLUDE", inclusion), ("EXCLUDE", exclusion)):
        if not MIN_PER_KIND <= len(items) <= MAX_PER_KIND:
            raise ValueError(f"{len(items)} critère(s) {name}, il en faut entre {MIN_PER_KIND} et {MAX_PER_KIND}")
    return Criteria(inclusion=inclusion, exclusion=exclusion, year_from=year_from, languages=list(languages))


def generate_criteria(llm, topic: str, questions, year_from=None, languages=("en",), trace=None):
    # generate_section attend une liste ; on enveloppe le Criteria dans une liste d'un élément.
    return generate_section(
        llm,
        build_criteria_prompt(topic, questions),
        lambda text: [parse_criteria(text, year_from, languages)],
        build_criteria_repair_prompt,
        trace=trace,
        section="criteria",
    )