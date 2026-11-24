"""Generation des questions de recherche : le LLM PROPOSE, le programme CONTROLE.

Reprend les acquis des TD01 et TD02 (prompt few-shot, regex stricte, un seul repair,
abstention vers l'humain), mais avec LLMClient (budget, retry, trace) et un nombre de
questions configurable.
"""
import re

from slrresearch.core.llm import LLMClient
from slrresearch.core.schemas.protocol import QuestionsOutcome, ResearchQuestion
from slrresearch.core.trace import Trace

RQ_PATTERN = re.compile(r"^(RQ[0-9]+)\.\s*(.{15,})$")
MAX_QUESTIONS = 5

_SYSTEM_RULES = (
    "Tu es un assistant méthodologique spécialisé dans les revues systématiques "
    "de la littérature (SLR). Reste dans le périmètre du sujet fourni. "
    "N'invente ni référence ni résultat scientifique. Réponds en français."
)

# Deux exemples de 5 lignes : on en montre les n premières (n <= 5).
_EXAMPLES = [
    (
        "IA et éducation",
        [
            "Comment l'IA influence-t-elle la personnalisation pédagogique ?",
            "Quels risques l'IA introduit-elle dans l'évaluation des étudiants ?",
            "Quel contrôle humain doit-il être conservé dans les décisions pédagogiques ?",
            "Quelles données les systèmes d'IA éducatifs utilisent-ils ?",
            "Quels effets mesurés sur la réussite des étudiants sont rapportés ?",
        ],
    ),
    (
        "DevOps et qualité",
        [
            "Comment DevOps influence-t-il la fréquence des déploiements ?",
            "Quels facteurs réduisent les défauts en production ?",
            "Quelles métriques évaluent cette amélioration ?",
            "Quels outils d'automatisation sont les plus utilisés ?",
            "Quelles limites organisationnelles freinent l'adoption ?",
        ],
    ),
]


def _check_n(n: int) -> None:
    if not 1 <= n <= MAX_QUESTIONS:
        raise ValueError(f"le nombre de questions doit etre entre 1 et {MAX_QUESTIONS}")


def _example_block(n: int) -> str:
    parts = []
    for i, (topic, questions) in enumerate(_EXAMPLES, start=1):
        lines = "\n".join(f"RQ{k}. {q}" for k, q in enumerate(questions[:n], start=1))
        parts.append(f"Exemple {i} - Sujet : {topic}\n{lines}")
    return "\n\n".join(parts)


def build_prompt(topic: str, n: int = 3) -> str:
    """Prompt few-shot (regles + exemples + sujet) en un seul texte."""
    _check_n(n)
    return f"""{_SYSTEM_RULES}

Propose exactement {n} questions de recherche ouvertes, distinctes et centrées sur le sujet.
N'ajoute ni introduction, ni conclusion, ni JSON.

{_example_block(n)}

Sujet courant : {topic.strip()}
Retourne exactement {n} lignes, de RQ1. à RQ{n}."""


def build_repair_prompt(text: str, error: str, n: int = 3) -> str:
    """Demande au LLM de corriger UNIQUEMENT la forme."""
    _check_n(n)
    first_example = "\n".join(f"RQ{k}. {q}" for k, q in enumerate(_EXAMPLES[0][1][:n], start=1))
    return f"""Tu corriges la forme d'une sortie destinée à un programme.

[TÂCHE]
Corrige uniquement la forme du texte. Préserve le sens des {n} questions.

[ERREUR DÉTECTÉE]
{error}

[CONTRAT ATTENDU]
- Exactement {n} lignes.
- Chaque ligne commence par RQ1., RQ2., ... dans l'ordre.
- N'ajoute ni introduction, ni explication, ni Markdown, ni JSON.

[EXEMPLE CONFORME]
{first_example}

[TEXTE À CORRIGER]
{text}

[SORTIE]
"""


def parse_questions(text: str, n: int = 3) -> list[ResearchQuestion]:
    """Controle deterministe. Leve ValueError (message precis) si le texte n'est pas conforme."""
    _check_n(n)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Texte vide")

    records = []
    for line in lines:
        match = RQ_PATTERN.fullmatch(line)
        if not match:
            raise ValueError(f"Ligne non conforme : {line}")
        records.append({"identifier": match.group(1), "question": match.group(2).strip()})

    expected = [f"RQ{i}" for i in range(1, n + 1)]
    if [r["identifier"] for r in records] != expected:
        raise ValueError(f"Identifiants attendus : {', '.join(expected)}")

    normalized = [" ".join(r["question"].lower().split()) for r in records]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Les questions doivent etre differentes")

    # pydantic.ValidationError herite de ValueError : meme chemin d'erreur que ci-dessus
    return [ResearchQuestion.model_validate(r) for r in records]


def generate_research_questions(
    llm: LLMClient,
    topic: str,
    n_questions: int = 3,
    trace: Trace | None = None,
) -> QuestionsOutcome:
    """PROPOSE (LLM) -> VALIDATE (regex+Pydantic) -> un seul REPAIR -> sinon HUMAN.

    Les erreurs d'appel (LLMError, BudgetExceeded) ne sont PAS attrapees ici : elles
    remontent, car ce n'est pas un probleme de forme mais un probleme d'infrastructure.
    """
    clean = topic.strip()
    if not clean:
        raise ValueError("Sujet vide")
    _check_n(n_questions)

    def log(event: str, **data) -> None:
        if trace:
            trace.log("protocol", event, **data)

    text = llm.call(build_prompt(clean, n_questions)).text.strip()
    first_error = ""  # Python supprime la variable de "except ... as e" a la fin du bloc : on la copie
    try:
        questions = parse_questions(text, n_questions)
    except ValueError as exc:
        first_error = str(exc)
        log("questions_invalid", error=first_error)
    else:
        log("questions_accepted", n=len(questions))
        return QuestionsOutcome(topic=clean, status="accepted", text=text, questions=questions)

    repaired = llm.call(build_repair_prompt(text, first_error, n_questions)).text.strip()
    try:
        questions = parse_questions(repaired, n_questions)
    except ValueError as exc:
        second_error = str(exc)
        log("questions_human_required", error=second_error)
        return QuestionsOutcome(
            topic=clean, status="human_required", text=repaired,
            error=second_error, repair_attempts=1,
        )
    log("questions_repaired", n=len(questions))
    return QuestionsOutcome(
        topic=clean, status="repaired", text=repaired, questions=questions, repair_attempts=1,
    )