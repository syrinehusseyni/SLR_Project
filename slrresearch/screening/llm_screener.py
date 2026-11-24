"""Agent B : le LLM propose un verdict, le programme le contrôle.

PROPOSE (un appel) -> VALIDATE (JSON + schéma + critères connus) -> un seul REPAIR -> sinon `invalid_format`.
Puis contrôle d'ancrage : la preuve citée doit apparaître dans le titre ou le résumé, sinon `uncertain`.
`BudgetExceeded` n'est jamais attrapée ici : elle remonte à l'étape.
"""
import hashlib
import json
import re
from pathlib import Path

from slrresearch.core.errors import ConfigError, LLMError
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.core.schemas.screening import LLMResult, ScreeningVerdict
from slrresearch.screening.text import normalize_for_matching

PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
PROMPT_NAME = "screening.txt"
LANGUAGE_ID = "LANG"  # identifiant réservé : exclusion pour langue non acceptée
MIN_EVIDENCE_CHARS = 12  # une « preuve » de quelques lettres serait trouvée partout


def load_prompt(name: str = PROMPT_NAME, directory: Path | None = None) -> str:
    path = (directory or PROMPT_DIR) / name
    if not path.exists():
        raise ConfigError(f"prompt introuvable : {path}")
    return path.read_text(encoding="utf-8")


def fingerprint(template: str, protocol: Protocol) -> str:
    """Empreinte de ce qui détermine les verdicts : si elle change, le cache n'est plus valable."""
    material = template + "\n" + protocol.topic + "\n" + protocol.criteria.model_dump_json()
    material += "\n" + json.dumps([q.model_dump() for q in protocol.questions], sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def known_criteria_ids(protocol: Protocol) -> set[str]:
    c = protocol.criteria
    return {item.id for item in c.inclusion + c.exclusion} | {LANGUAGE_ID}


def build_prompt(template: str, paper: Paper, protocol: Protocol) -> str:
    """Substitution en un seul passage : un titre contenant « {{topic}} » n'est pas ré-interprété."""
    c = protocol.criteria
    values = {
        "topic": protocol.topic,
        "questions": "\n".join(f"{q.identifier} : {q.question}" for q in protocol.questions),
        "inclusion": "\n".join(f"{i.id} : {i.text}" for i in c.inclusion),
        "exclusion": "\n".join(f"{e.id} : {e.text}" for e in c.exclusion),
        "year_from": str(c.year_from) if c.year_from is not None else "aucune",
        "languages": ", ".join(c.languages) if c.languages else "toutes",
        "title": paper.title,
        "abstract": paper.abstract,
    }

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise ConfigError(f"variable inconnue dans le prompt : {{{{{key}}}}}")
        return values[key]

    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def build_repair_prompt(prompt: str, previous: str, error: str) -> str:
    return (
        f"{prompt}\n\nTA RÉPONSE PRÉCÉDENTE ÉTAIT INVALIDE.\nRéponse : {previous[:1500]}\nProblème : {error}\n"
        "Réponds de nouveau par UN SEUL objet JSON valide, sans texte autour, sans markdown."
    )


def _extract_json(text: str) -> dict:
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


def parse_verdict(text: str, paper_id: str, known_ids: set[str]) -> ScreeningVerdict:
    """Lève ValueError si la réponse n'est pas exploitable (ValidationError en est une)."""
    data = _extract_json(text)
    reasons = data.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    criteria = data.get("criteria", [])
    if isinstance(criteria, str):
        criteria = [criteria]
    decision = str(data.get("decision", "")).strip().lower()
    if "confidence" not in data:
        raise ValueError("champ « confidence » manquant")
    verdict = ScreeningVerdict(
        paper_id=paper_id,  # l'identifiant appartient à l'application, pas au LLM
        decision=decision,
        confidence=data["confidence"],
        criteria=[str(x).strip() for x in criteria if str(x).strip()],
        reasons=[str(r).strip() for r in reasons if str(r).strip()],
        evidence=str(data.get("evidence", "") or "").strip(),
    )
    unknown = [cid for cid in verdict.criteria if cid not in known_ids]
    if unknown:
        raise ValueError(f"critères inconnus : {unknown} (autorisés : {sorted(known_ids)})")
    if verdict.decision != "uncertain":
        if not verdict.criteria:
            raise ValueError("include/exclude doit citer au moins un critère")
        if not verdict.evidence:
            raise ValueError("include/exclude doit fournir une preuve (evidence)")
    return verdict


def check_grounding(paper: Paper, verdict: ScreeningVerdict) -> None:
    """La preuve non vide doit apparaître dans titre + résumé (après normalisation)."""
    if not verdict.evidence.strip():
        if verdict.decision == "uncertain":
            return
        raise ValueError("preuve vide pour une décision définitive")
    evidence = normalize_for_matching(verdict.evidence)
    source = normalize_for_matching(f"{paper.title} {paper.abstract}")
    if len(evidence) < MIN_EVIDENCE_CHARS:
        raise ValueError("preuve trop courte pour être vérifiable")
    if evidence not in source:
        raise ValueError("preuve LLM absente du titre et du résumé")


def screen_paper(llm, paper: Paper, protocol: Protocol, template: str, trace=None) -> LLMResult:
    prompt = build_prompt(template, paper, protocol)
    known = known_criteria_ids(protocol)

    def log(event: str, **data):
        if trace is not None:
            trace.log("screening", event, paper_id=paper.id, **data)

    try:
        text = llm.call(prompt).text
    except LLMError as exc:
        note = str(exc)
        log("llm_error", error=note)
        return LLMResult(paper_id=paper.id, status="llm_error", note=note)

    try:
        verdict = parse_verdict(text, paper.id, known)
    except ValueError as exc:
        first_error = str(exc)
        log("repair", error=first_error)
        try:
            text = llm.call(build_repair_prompt(prompt, text, first_error)).text
        except LLMError as exc2:
            note = str(exc2)
            log("llm_error", error=note)
            return LLMResult(paper_id=paper.id, status="llm_error", note=note)
        try:
            verdict = parse_verdict(text, paper.id, known)
        except ValueError as exc3:
            note = str(exc3)
            log("invalid_format", error=note, raw=text[:500])  # texte brut conservé pour comprendre
            return LLMResult(paper_id=paper.id, status="invalid_format", note=note)

    try:
        check_grounding(paper, verdict)
    except ValueError as exc:
        note = f"Contrôle local : {exc} (décision proposée : {verdict.decision})"
        log("grounding_failed", error=note)
        downgraded = verdict.model_copy(update={
            "decision": "uncertain", "confidence": 0.0, "evidence": "",
            "reasons": [*verdict.reasons, note],
        })
        return LLMResult(paper_id=paper.id, status="grounding_failed", verdict=downgraded, note=note)

    log("llm_verdict", decision=verdict.decision, confidence=verdict.confidence)
    return LLMResult(paper_id=paper.id, status="ok", verdict=verdict)