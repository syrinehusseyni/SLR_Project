"""Fiche d'étude, un champ à la fois : le LLM propose, le programme contrôle.

Pour chaque champ : RETRIEVE (les chunks les plus proches, requête en anglais) -> PROPOSE (un appel LLM, réponse JSON)
-> VALIDATE (JSON + règles) -> VERIFY (la citation est retrouvée dans un des chunks montrés, la page vient de ce chunk, chaque
nombre et chaque nom propre de `value` figure dans la citation, la citation parle de l'étude des auteurs) -> UN SEUL REPAIR, qu'il s'agisse
d'un JSON invalide ou d'un contrôle raté (l'erreur précise est renvoyée au LLM) -> sinon `invalid_format` ou `rejected`.
`BudgetExceeded` n'est jamais attrapée ici : elle remonte à l'étape.
ATTENTION : `verified` veut dire « citation retrouvée, nombres et noms propres de la valeur présents dans la citation ». Ce n'est PAS
une preuve que la citation soutient toute la valeur : le juge de fidélité (modèle différent) viendra à l'étape 6.
"""
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from slrresearch.core.errors import ConfigError, LLMError
from slrresearch.core.schemas.extraction import Chunk
from slrresearch.core.schemas.study_card import CardField, StudyCard
from slrresearch.download.agent import safe_filename
from slrresearch.extraction.retrieval import lexical_retrieve

PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
PROMPT_NAME = "study_card.txt"
MIN_QUOTE_CHARS = 20   # une citation de quelques mots serait trouvée partout
MAX_QUOTE_CHARS = 400  # sinon le LLM recopie le chunk entier et « vérifie » tout
MAX_VALUE_CHARS = 500
VERIFY_VERSION = "3"  # à changer quand les règles de contrôle changent : les fiches sont alors refaites
DIRNAME = "study_cards"


@dataclass(frozen=True)
class FieldSpec:
    id: str
    label: str
    instruction: str  # en français : lu par le LLM
    query: str        # en anglais : sert à retrouver les chunks dans des articles anglais


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("design", "Type d'étude", "type d'étude et protocole (expérience comparative, revue, enquête, outil proposé...)",
              "study design methodology we evaluate experiment experiments approach propose compare benchmark review"),
    FieldSpec("data", "Données", "données ou échantillon utilisés (jeux de données, nombre d'articles ou d'études, domaine)",
              "dataset datasets corpus data samples size number abstracts articles reviews domain"),
    FieldSpec("models", "Modèles", "modèles ou méthodes utilisés ou évalués",
              "model models GPT LLaMA BERT ChatGPT Claude Gemini prompt prompting fine-tuned zero-shot"),
    FieldSpec("findings", "Résultats", "résultats principaux rapportés, avec les chiffres s'il y en a",
              "results recall precision accuracy F1 sensitivity specificity performance outperform achieved"),
    FieldSpec("limitations", "Limites", "limites ou risques signalés par les auteurs",
              "limitations limitation future work however threats bias risk hallucination reproducibility"),
)


# -- prompt ----------------------------------------------------------------------------------------------------
def load_prompt(name: str = PROMPT_NAME, directory: Path | None = None) -> str:
    path = (directory or PROMPT_DIR) / name
    if not path.exists():
        raise ConfigError(f"prompt introuvable : {path}")
    return path.read_text(encoding="utf-8")


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[CHUNK {c.chunk_id}] [PAGE {c.page}]\n{c.text}" for c in chunks)


def build_prompt(template: str, spec: FieldSpec, chunks: list[Chunk]) -> str:
    """Substitution en un seul passage : un extrait contenant « {{label}} » n'est pas ré-interprété."""
    values = {"label": spec.label, "instruction": spec.instruction, "context": build_context(chunks)}

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise ConfigError(f"variable inconnue dans le prompt : {{{{{key}}}}}")
        return values[key]

    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def build_repair_prompt(prompt: str, previous: str, error: str) -> str:
    json_hint = (" Dans une chaîne JSON, un guillemet double doit s'écrire \\\" et un retour à la ligne \\n ; recopie la citation sur une seule ligne."
                 if error.startswith("JSON invalide") else "")
    return (f"{prompt}\n\nTA RÉPONSE PRÉCÉDENTE ÉTAIT INVALIDE.\nRéponse : {previous[:1500]}\nProblème : {error}.{json_hint}\n"
            "Corrige uniquement ce problème : recopie la citation exactement depuis un extrait, et n'écris dans value que ce que dit la citation. "
            "Si tu ne peux pas, réponds {\"status\": \"not_reported\"}. "
            "Réponds par UN SEUL objet JSON valide, sans texte autour, sans markdown.")


# -- réponse ---------------------------------------------------------------------------------------------------
@dataclass
class Answer:
    status: str  # found | not_reported
    value: str = ""
    chunk_id: str = ""
    quote: str = ""
    scope: str = "own_study"  # own_study | other (travaux cités, annexe, déclaration d'usage d'IA...)


def _extract_json(text: str) -> dict:
    """Même logique que dans llm_screener : le premier objet JSON de la réponse, même entouré de texte ou de ```."""
    if not text or not text.strip():
        raise ValueError("réponse vide")
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("aucun objet JSON trouvé")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide : {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("la réponse doit être un objet JSON")
    return data


def _clean_chunk_id(raw: str) -> str:
    cid = str(raw or "").strip().strip("[]").strip()
    return cid[len("CHUNK "):].strip() if cid.upper().startswith("CHUNK ") else cid


def parse_answer(text: str) -> Answer:
    """Lève ValueError si la réponse n'est pas exploitable."""
    data = _extract_json(text)
    status = str(data.get("status", "")).strip().lower()
    if status == "not_reported":
        return Answer("not_reported")
    if status != "found":
        raise ValueError('« status » doit valoir "found" ou "not_reported"')
    value = str(data.get("value", "") or "").strip()
    chunk_id = _clean_chunk_id(data.get("chunk_id", ""))
    quote = str(data.get("quote", "") or "").strip()
    if not value or not chunk_id or not quote:
        raise ValueError('"found" exige value, chunk_id et quote non vides')
    if len(value) > MAX_VALUE_CHARS:
        raise ValueError(f"value trop longue ({len(value)} caractères, maximum {MAX_VALUE_CHARS})")
    scope = str(data.get("scope", "own_study") or "own_study").strip().lower()  # absent : traité comme own_study
    if scope not in ("own_study", "other"):
        raise ValueError('« scope » doit valoir "own_study" ou "other"')
    return Answer("found", value, chunk_id, quote, scope)


# -- contrôle --------------------------------------------------------------------------------------------------
_TYPOGRAPHY = {
    **dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-"),   # tirets et traits d'union typographiques
    **dict.fromkeys(map(ord, "\u2018\u2019\u201a\u2032"), "'"),
    **dict.fromkeys(map(ord, "\u201c\u201d\u201e\u2033"), '"'),
    **dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007"), " "),
    **dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None),                # trait d'union conditionnel, espaces de largeur nulle
}
_LETTER_HYPHEN_LETTER = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")


def flat(text: str) -> str:
    """Comparaison tolérante MAIS sans effet sur le contenu : espaces, casse, typographie (tirets, apostrophes, guillemets) et
    trait d'union entre deux lettres sont ignorés. Un trait d'union entre chiffres est conservé (« 1-2 » n'est pas « 12 »)."""
    text = text.translate(_TYPOGRAPHY)
    text = _LETTER_HYPHEN_LETTER.sub("", text)   # « chain-of-thought » = « chain-ofthought » (césure ambiguë en fin de ligne)
    return " ".join(text.split()).casefold()


_NUMBER = re.compile(r"[1-9]\d{0,2}(?:[ ,]\d{3})+(?:\.\d+)?(?!\d)|\d+(?:[.,]\d+)?")   # un groupe de milliers ne commence pas par 0
_THOUSANDS = re.compile(r"[1-9]\d{0,2}(?:[ ,]\d{3})+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Nombres d'un texte. « 0,82 » = « 0.82 » ; « 8,124 » = « 8 124 » = « 8124 » (et on garde aussi « 8 » et « 124 », car
    dans un tableau « 108 780 » peut être deux nombres). « 0,008 » est un décimal (un groupe de milliers ne commence pas par 0).
    Limite : « 82 % » n'est pas « 0.82 »."""
    text = text.translate(_TYPOGRAPHY)
    found: set[str] = set()
    for match in _NUMBER.findall(text):
        if _THOUSANDS.fullmatch(match):
            found.add(re.sub(r"[ ,]", "", match))
            found.update(part for part in re.split(r"[ ,]", match) if part)
        else:
            found.add(match.replace(",", "."))
    return found


_IGNORED_TERMS = {"IA", "LLM", "LLMs"}  # sigles que le LLM écrit en français sans qu'ils figurent dans une citation anglaise
_PROMPT_TRANSLATIONS = {"rappel", "precision", "exactitude", "sensibilite", "specificite"}  # demandés par le prompt : jamais suspects


def _fold(text: str) -> str:
    """Casse et accents ignorés (« Précision » = « precision »), pour comparer un mot de la valeur à la citation."""
    return "".join(c for c in unicodedata.normalize("NFKD", flat(text)) if not unicodedata.combining(c))


def unsupported_terms(value: str, quote: str) -> list[str]:
    """Noms propres, sigles et termes à chiffre de `value` (mots avec majuscule ou chiffre, sauf le premier mot de la valeur et ceux
    qui commencent une phrase) qui ne figurent pas dans la citation. Un nom de modèle ou de jeu de données qui n'est pas dans la
    citation est presque toujours une invention ou un emprunt à une autre partie du texte. Les nombres seuls relèvent de numbers_in."""
    source = _fold(quote)
    missing: list[str] = []
    previous = ""
    for position, raw in enumerate(value.split()):
        sentence_start = position == 0 or previous.endswith((".", "!", "?"))
        previous = raw
        for index, part in enumerate(re.split(r"[=/]", raw)):  # « Rappel=1.0 » = deux mots
            token = part.strip("()[]{}«»\"'“”‘’,;:!?").strip(".-")
            if (sentence_start and index == 0) or len(token) < 2 or token in _IGNORED_TERMS or re.fullmatch(r"[\d.,\-%]+", token):
                continue
            if _fold(token) in _PROMPT_TRANSLATIONS:
                continue
            if (token[0].isupper() or any(ch.isdigit() for ch in token)) and _fold(token) not in source and token not in missing:
                missing.append(token)
    return missing


def verify_answer(answer: Answer, spec: FieldSpec, shown: list[Chunk]) -> CardField:
    """Contrôles programme : le LLM ne décide jamais seul qu'une information est établie."""
    base = {"field_id": spec.id, "label": spec.label}
    if answer.status == "not_reported":
        return CardField(**base, status="not_reported")
    if answer.scope == "other":  # le LLM dit que le passage parle d'autre chose que de l'étude des auteurs
        return CardField(**base, status="not_reported", note="passage relatif à un autre travail, à une annexe ou à une déclaration d'usage (scope=other)")
    detail = {"value": answer.value, "chunk_id": answer.chunk_id, "quote": answer.quote}
    quote_flat = flat(answer.quote)
    cited = next((c for c in shown if c.chunk_id == answer.chunk_id), None)
    chunk, note = None, ""
    if cited is not None and quote_flat in flat(cited.text):
        chunk = cited
    else:  # un petit modèle se trompe souvent d'identifiant : la citation fait foi, pas l'identifiant
        chunk = next((c for c in shown if quote_flat in flat(c.text)), None)
        if chunk is not None:
            note = f"chunk_id corrigé : le LLM citait {answer.chunk_id!r}, la citation est dans {chunk.chunk_id}"
            detail["chunk_id"] = chunk.chunk_id
    problem = None
    if len(quote_flat) < MIN_QUOTE_CHARS:
        problem = f"citation trop courte (moins de {MIN_QUOTE_CHARS} caractères)"
    elif len(answer.quote) > MAX_QUOTE_CHARS:
        problem = f"citation trop longue ({len(answer.quote)} caractères, maximum {MAX_QUOTE_CHARS}) : choisis UNE phrase de 300 caractères au plus"
    elif chunk is None and cited is None:
        problem = "chunk_id absent des extraits fournis et citation absente des extraits : recopie-la exactement depuis l'un d'eux"
    elif chunk is None:
        problem = "citation absente du chunk cité (et des autres extraits) : recopie-la exactement"
    elif missing := sorted(numbers_in(answer.value) - numbers_in(answer.quote)):
        problem = f"nombres de la valeur absents de la citation : {missing}"
    elif terms := unsupported_terms(answer.value, answer.quote):
        problem = f"termes de la valeur absents de la citation : {terms}"
    if problem:
        return CardField(**base, status="rejected", note=problem, **detail)
    return CardField(**base, status="verified", page=chunk.page, note=note, **detail)  # la page vient du chunk, jamais du LLM


def fill_field(llm, spec: FieldSpec, chunks: list[Chunk], template: str, top_k: int) -> CardField:
    """Un champ de la fiche : au plus 2 appels (essai + une réparation, que l'erreur soit un JSON invalide ou un contrôle raté).
    Ne lève que `BudgetExceeded` (ou une erreur de programmation)."""
    base = {"field_id": spec.id, "label": spec.label}
    shown = [chunk for chunk, _ in lexical_retrieve(chunks, spec.query, top_k)]
    if not shown:
        return CardField(**base, status="no_context", note="aucun extrait ne ressemble à la requête de recherche")
    prompt = build_prompt(template, spec, shown)
    available = ", ".join(c.chunk_id for c in shown)
    text, problem = "", ""
    for attempt in (1, 2):
        try:
            text = llm.call(prompt if attempt == 1 else build_repair_prompt(prompt, text, problem)).text
        except LLMError as exc:
            return CardField(**base, status="llm_error", note=str(exc))
        try:
            answer = parse_answer(text)
        except ValueError as exc:
            if attempt == 2:
                if not text.strip():  # réponse vide (modèle de raisonnement à court de tokens...) : panne, retentée à la reprise
                    return CardField(**base, status="llm_error", note="réponse vide du modèle après une nouvelle tentative")
                return CardField(**base, status="invalid_format", note=str(exc))
            problem = str(exc)
            continue
        field = verify_answer(answer, spec, shown)
        if field.status != "rejected" or attempt == 2:
            return field
        problem = f"{field.note}. Extraits disponibles : {available}"
    raise AssertionError("inatteignable")  # pragma: no cover


# -- reprise : empreinte et fichier par papier -----------------------------------------------------------------
def chunks_digest(chunks: list[Chunk]) -> str:
    return hashlib.sha256("\n".join(f"{c.chunk_id}\n{c.text}" for c in chunks).encode("utf-8")).hexdigest()


def fingerprint(template: str, top_k: int, model: str, chunks: list[Chunk]) -> str:
    """Ce qui détermine la fiche : si l'un de ces éléments change, la fiche n'est plus valable."""
    material = "\n".join([template, str(top_k), model, VERIFY_VERSION, chunks_digest(chunks),
                          json.dumps([(f.id, f.label, f.instruction, f.query) for f in FIELDS])])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def card_file(run_dir: Path, paper_id: str) -> Path:
    return run_dir / DIRNAME / (safe_filename(paper_id) + ".json")


def save_card(path: Path, card: StudyCard) -> None:
    """Écriture atomique (fichier temporaire puis os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(card.model_dump(mode="json"), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(partial, path)


def load_card(path: Path) -> StudyCard | None:
    try:
        return StudyCard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None