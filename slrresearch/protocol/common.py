"""Outils communs aux trois générations de l'étape 1b : concepts, critères, facettes.

Même principe que pour les questions (étape 1a) :
PROPOSE -> VALIDATE -> un seul REPAIR -> sinon human_required.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# Caractères typographiques que certains LLM produisent (tirets spéciaux, guillemets courbes...).
_REPLACEMENTS = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}

_NUMBERING = re.compile(r"^\d+[.)]\s+")


def normalize_text(text: str) -> str:
    """NFKC + remplacement des tirets/guillemets typographiques par leurs équivalents simples."""
    text = unicodedata.normalize("NFKC", text)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def clean_lines(text: str) -> list[str]:
    """Lignes non vides, normalisées : sans gras/italique markdown, sans puces ni numérotation."""
    lines: list[str] = []
    for raw in normalize_text(text).splitlines():
        line = raw.replace("**", "").replace("__", "").replace("`", "").strip()
        if not line:
            continue
        if line[:2] in ("- ", "* "):
            line = line[2:].strip()
        line = _NUMBERING.sub("", line)
        if line:
            lines.append(line)
    return lines


@dataclass
class SectionResult:
    status: Literal["accepted", "repaired", "human_required"]
    items: list[Any] = field(default_factory=list)
    text: str = ""
    error: str = ""
    repair_attempts: int = 0


def generate_section(
    llm,
    prompt: str,
    parse: Callable[[str], list],
    build_repair: Callable[[str, str], str],
    trace=None,
    section: str = "section",
) -> SectionResult:
    """PROPOSE -> VALIDATE -> un seul REPAIR. `LLMError` et `BudgetExceeded` remontent."""
    text = llm.call(prompt).text
    try:
        items = parse(text)
    except ValueError as exc:
        first_error = str(exc)  # copié tout de suite : `exc` disparaît à la fin du bloc except
    else:
        _log(trace, section, "accepted")
        return SectionResult("accepted", items, text)

    _log(trace, section, "invalid", error=first_error, text=text)
    repaired_text = llm.call(build_repair(text, first_error)).text
    try:
        items = parse(repaired_text)
    except ValueError as exc:
        second_error = str(exc)
    else:
        _log(trace, section, "repaired")
        return SectionResult("repaired", items, repaired_text, repair_attempts=1)

    _log(trace, section, "human_required", error=second_error, text=repaired_text)
    return SectionResult("human_required", [], repaired_text, second_error, repair_attempts=1)


def _log(trace, section: str, outcome: str, **data) -> None:
    if trace is not None:
        trace.log("protocol", f"{section}_{outcome}", **data)


def questions_block(questions) -> str:
    return "\n".join(f"{q.identifier}. {q.question}" for q in questions)