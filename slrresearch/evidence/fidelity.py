"""Étape 6a : juge LLM distinct pour les champs vérifiés des fiches d'étude."""
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ValidationError

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded, LLMError
from slrresearch.core.schemas.extraction import ExtractionOutcome, PaperText
from slrresearch.core.schemas.fidelity import FidelityEntry, FidelityOutcome, FidelityReport, FidelitySummary
from slrresearch.core.schemas.study_card import StudyCardsOutcome, StudyCard
from slrresearch.extraction.reader import load_paper_text
from slrresearch.extraction.study_card import flat, numbers_in
from slrresearch.hitl.queue import ReviewItem

ARTIFACT = "fidelity"
VERSION = "fidelity-judge-v2"
ROLE = "judge"
PROMPT_NAME = "fidelity_judge.txt"


class JudgeAnswer(BaseModel):
    entailment: str
    own_study: bool
    reason: str
    unsupported_claims: list[str]


def _prompt(template: str, field_id: str, label: str, value: str, quote: str, chunk_text: str) -> str:
    context = json.dumps({"field_id": field_id, "label": label, "value": value,
                          "quote": quote, "chunk_text": chunk_text}, ensure_ascii=False)
    return template.replace("{{context}}", context)


def _parse(text: str) -> JudgeAnswer:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("aucun objet JSON")
    answer = JudgeAnswer.model_validate_json(text[start:end + 1])
    if answer.entailment not in ("yes", "partial", "no"):
        raise ValueError("entailment doit être yes, partial ou no")
    return answer


def _fingerprint(judge_model: str, card: StudyCard, text: PaperText, prompt: str, field) -> str:
    material = json.dumps([VERSION, judge_model, card.fingerprint, text.source_sha256, prompt,
                           field.field_id, field.value, field.chunk_id, field.page, field.quote], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


_KNOWN_CLAIM_TERMS = {
    "étude comparative": ("compar",),
    "expérience comparative": ("compar",),
    "expérimentale": ("experiment",),
    "expérience": ("experiment",),
    "entraînement": ("training", "trained", "train ", "fine-tun"),
    "validation croisée": ("cross-validation", "cross validation", "crossvalidated"),
}


def _unsupported_known_claims(value: str, quote: str) -> list[str]:
    """Catch a few high-risk translated method claims when the cited sentence omits them.

    These checks are conservative: they route a judge's apparent `yes` to human review;
    they never reject solely because wording differs.
    """
    value_fold, quote_fold = _fold(value), _fold(quote)
    missing = []
    for claim, aliases in _KNOWN_CLAIM_TERMS.items():
        if _fold(claim) in value_fold and not any(alias in quote_fold for alias in aliases):
            missing.append(claim)
    return missing


def _report(paper_id: str, field, chunk, judge_model: str, client, template: str, input_fingerprint: str) -> FidelityReport:
    in_chunk = bool(chunk and flat(field.quote) in flat(chunk.text))
    provenance = bool(chunk and chunk.paper_id == paper_id and chunk.page == field.page)
    numbers_ok = numbers_in(field.value) <= numbers_in(field.quote)
    common = dict(paper_id=paper_id, field_id=field.field_id, value=field.value,
                  chunk_id=field.chunk_id, page=field.page, quote=field.quote,
                  quote_in_chunk=in_chunk, provenance_ok=provenance, numbers_ok=numbers_ok,
                  judge_model=judge_model, input_fingerprint=input_fingerprint)
    if not in_chunk or not provenance or not numbers_ok:
        problems = []
        if not in_chunk: problems.append("citation absente du chunk")
        if not provenance: problems.append("provenance papier/page incohérente")
        if not numbers_ok: problems.append("nombre de la valeur absent de la citation")
        return FidelityReport(**common, entailment="no", own_study=False,
                              reason="; ".join(problems), status="rejected")
    prompt = _prompt(template, field.field_id, field.label, field.value, field.quote, chunk.text)
    missing_claims = _unsupported_known_claims(field.value, field.quote)
    last_error = ""
    for attempt in range(2):
        try:
            answer = _parse(client.call(prompt).text)
            entailment = answer.entailment
            reason = answer.reason
            if missing_claims and entailment == "yes":
                entailment = "partial"
                reason = ("La citation ne contient pas de termes appuyant : " + ", ".join(missing_claims)
                          + (f". Avis du juge : {answer.reason}" if answer.reason else ""))
            if entailment == "yes" and answer.own_study:
                status = "verified"
            elif entailment == "no" or not answer.own_study:
                status = "rejected"
            else:
                status = "human_review"
            return FidelityReport(**common, entailment=entailment, own_study=answer.own_study,
                                  reason=reason, status=status)
        except LLMError as exc:
            return FidelityReport(**common, entailment="error", reason=str(exc), status="llm_error")
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
    return FidelityReport(**common, entailment="error", reason=f"réponse juge invalide : {last_error}", status="llm_error")


class FidelityStage:
    name = ARTIFACT

    def __init__(self, max_papers=None, pause=0.0, force=False, prompt_dir: Path | None = None, sleep=time.sleep):
        if max_papers is not None and max_papers <= 0: raise ValueError("max_papers doit être > 0")
        if pause < 0: raise ValueError("pause doit être >= 0")
        self.max_papers, self.pause, self.force = max_papers, pause, force
        self.prompt_dir, self._sleep = prompt_dir, sleep

    def run(self, ctx) -> StageResult:
        if not ctx.settings.judge_is_independent:
            return StageResult(status="failed", message=("Le juge doit utiliser un modèle différent de l'extracteur. "
                "Configurez GROQ_MODEL_JUDGE ou OLLAMA_MODEL_JUDGE, puis relancez."))
        try:
            cards_index = ctx.store.read_artifact("study_cards", StudyCardsOutcome)
            extraction = ctx.store.read_artifact("extraction", ExtractionOutcome)
        except ArtifactNotFound as exc:
            return StageResult(status="failed", message=f"artefact requis introuvable : {exc}")
        if self.max_papers is not None and len(cards_index.papers) > self.max_papers:
            return StageResult(status="failed", message=f"{len(cards_index.papers)} fiches dépassent --max-papers {self.max_papers}; aucun appel fait")
        prompt_path = self.prompt_dir / PROMPT_NAME if self.prompt_dir else Path("configs/prompts") / PROMPT_NAME
        try:
            template = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            return StageResult(status="failed", message=f"prompt juge introuvable : {prompt_path} ({exc})")
        judge_model = ctx.settings.model_for(ROLE)
        extraction_files = {p.paper_id: p.file for p in extraction.papers if p.status == "extracted"}
        client = ctx.llm(ROLE)
        previous = {}
        if not self.force and ctx.store.exists(ARTIFACT):
            try:
                old = ctx.store.read_artifact(ARTIFACT, FidelityOutcome)
                previous = {(r.paper_id, r.field_id): r for p in old.papers for r in p.reports
                            if old.judge_model == judge_model and old.version == VERSION and r.status != "llm_error"}
            except Exception:
                previous = {}
        papers, review_items = [], []
        consecutive_errors = 0
        provider_problem = ""
        for entry in cards_index.papers:
            if entry.status == "failed" or entry.paper_id not in extraction_files:
                continue
            try:
                card_path = ctx.store.run_dir / entry.file
                card = StudyCard.model_validate_json(card_path.read_text(encoding="utf-8"))
                text = load_paper_text(ctx.store.run_dir / extraction_files[entry.paper_id])
                if text is None:
                    continue
                chunks = {c.chunk_id: c for c in text.chunks}
                reports = []
                input_fp = None
                calls_before_paper = client.calls
                for field in card.fields:
                    input_fp = _fingerprint(judge_model, card, text, template, field)
                    cached = previous.get((card.paper_id, field.field_id))
                    if cached is not None and cached.input_fingerprint == input_fp:
                        cached = _apply_human_decision(cached, ctx.queue)
                        reports.append(cached)
                        if cached.status == "human_review":
                            item = ctx.queue.open_item("G5", "evidence", f"{card.paper_id}:{field.field_id}",
                                f"Juge fidélité partial : {cached.reason}")
                            if item.status == "open": review_items.append(item)
                        continue
                    if field.status != "verified":
                        reports.append(FidelityReport(paper_id=card.paper_id, field_id=field.field_id, value=field.value,
                            chunk_id=field.chunk_id, page=field.page, quote=field.quote, quote_in_chunk=False,
                            provenance_ok=False, numbers_ok=False, entailment="not_checked", judge_model=judge_model,
                            input_fingerprint=input_fp,
                            reason=f"champ source: {field.status}", status="skipped"))
                        continue
                    chunk = chunks.get(field.chunk_id)
                    if provider_problem:
                        report = _report_without_call(card.paper_id, field, chunk, judge_model, input_fp, provider_problem)
                    else:
                        if self.pause and client.calls > calls_before_paper:
                            self._sleep(self.pause)
                        try:
                            report = _report(card.paper_id, field, chunk, judge_model, client, template, input_fp)
                        except BudgetExceeded as exc:
                            provider_problem = f"budget atteint : {exc}"
                            report = _report_without_call(card.paper_id, field, chunk, judge_model, input_fp, provider_problem)
                    reports.append(report)
                    if report.status == "llm_error":
                        consecutive_errors += 1
                        if consecutive_errors >= 2:
                            provider_problem = report.reason or "deux erreurs LLM consécutives; arrêt précoce"
                    else:
                        consecutive_errors = 0
                    if report.status == "human_review":
                        item = ctx.queue.open_item("G5", "evidence", f"{card.paper_id}:{field.field_id}",
                            f"Juge fidélité partial : {report.reason}")
                        if item.status == "open": review_items.append(item)
                    ctx.trace.log("fidelity_judge", "field_done", paper_id=card.paper_id, field=field.field_id,
                                  status=report.status, entailment=report.entailment)
                papers.append(FidelityEntry(paper_id=card.paper_id, file=entry.file, reports=reports))
            except (OSError, ValueError, ValidationError) as exc:
                ctx.trace.log("fidelity_judge", "paper_error", paper_id=entry.paper_id, error=f"{type(exc).__name__}: {exc}")
                continue
        flat_reports = [r for p in papers for r in p.reports]
        counts = {status: sum(r.status == status for r in flat_reports) for status in ("verified", "human_review", "rejected", "llm_error", "skipped")}
        outcome = FidelityOutcome(version=VERSION, judge_model=judge_model, papers=papers,
            summary=FidelitySummary(papers=len(papers), **counts, llm_calls=client.calls, tokens=client.total_tokens))
        ctx.store.write_artifact(ARTIFACT, FidelityOutcome, outcome)
        ctx.trace.log("fidelity_judge", "written", **outcome.summary.model_dump())
        if review_items:
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=review_items,
                               message=f"{len(review_items)} champs attendent une revue G5")
        if counts["llm_error"]:
            return StageResult(status="failed", artifact=ARTIFACT, message=(
                f"Juge interrompu après {client.calls} appels ({counts['llm_error']} champs en erreur). "
                "Les erreurs sont conservées et seront retentées à la prochaine exécution."))
        return StageResult(status="done", artifact=ARTIFACT)


def _report_without_call(paper_id: str, field, chunk, judge_model: str, input_fingerprint: str, reason: str) -> FidelityReport:
    in_chunk = bool(chunk and flat(field.quote) in flat(chunk.text))
    provenance = bool(chunk and chunk.paper_id == paper_id and chunk.page == field.page)
    numbers_ok = numbers_in(field.value) <= numbers_in(field.quote)
    return FidelityReport(paper_id=paper_id, field_id=field.field_id, value=field.value, chunk_id=field.chunk_id,
        page=field.page, quote=field.quote, quote_in_chunk=in_chunk, provenance_ok=provenance,
        numbers_ok=numbers_ok, entailment="error", own_study=None, reason=reason, judge_model=judge_model,
        input_fingerprint=input_fingerprint, status="llm_error")


def _apply_human_decision(report: FidelityReport, queue, item_id: str | None = None) -> FidelityReport:
    if report.status != "human_review":
        return report
    item = queue.get(item_id or f"G5:evidence:{report.paper_id}:{report.field_id}")
    if item is None or item.status != "closed" or item.decision not in ("approved", "rejected"):
        return report
    return report.model_copy(update={
        "status": "verified" if item.decision == "approved" else "rejected",
        "human_decision": item.decision,
        "reviewer": item.decided_by,
    })
