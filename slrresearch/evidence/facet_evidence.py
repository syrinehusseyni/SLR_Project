"""Step 6b: reuse approved study-card evidence and extract only uncovered facet categories."""
import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded
from slrresearch.core.schemas.extraction import ExtractionOutcome
from slrresearch.core.schemas.facet_evidence import (
    FacetEvidenceOutcome, FacetEvidencePaper, FacetEvidenceRecord, FacetEvidenceSummary,
)
from slrresearch.core.schemas.facet_plan import FacetPlanOutcome
from slrresearch.core.schemas.fidelity import FidelityOutcome, FidelityReport
from slrresearch.core.schemas.study_card import StudyCardsOutcome, StudyCard
from slrresearch.evidence.fidelity import _apply_human_decision, _report, PROMPT_NAME as FIDELITY_PROMPT_NAME
from slrresearch.extraction.reader import load_paper_text
from slrresearch.extraction.study_card import FieldSpec, fill_field, load_prompt

ARTIFACT = "facet_evidence"
VERSION = "facet-evidence-v2"

_SUPPLEMENTAL_FIELDS = {
    "parameters": FieldSpec(
        "parameters", "Paramètres", "paramètres d'entraînement, de prompting ou de configuration rapportés par les auteurs",
        "hyperparameters prompt settings temperature training epochs fine-tuning configuration",
    ),
    "methodology": FieldSpec(
        "methodology", "Méthodologie et pratiques", "méthodes d'intégration du LLM au workflow de screening et pratiques recommandées",
        "methodology workflow integration implementation human in the loop guidelines best practices screening",
    ),
}


class _CallPacer:
    """One rate limiter shared by both LLM roles in this stage."""

    def __init__(self, seconds: float, sleep, clock):
        self.seconds, self.sleep, self.clock = seconds, sleep, clock
        self.last_call = None

    def call(self, client, prompt: str):
        if self.seconds and self.last_call is not None:
            remaining = self.seconds - (self.clock() - self.last_call)
            if remaining > 0:
                self.sleep(remaining)
        try:
            return client.call(prompt)
        finally:
            self.last_call = self.clock()


class _PacedClient:
    def __init__(self, client, pacer: _CallPacer):
        self.client, self.pacer = client, pacer

    def call(self, prompt: str):
        return self.pacer.call(self.client, prompt)


def _fingerprint(plan: FacetPlanOutcome, paper_text, extraction_model: str, judge_model: str,
                 extraction_template: str, judge_template: str, facet, spec: FieldSpec, top_k: int) -> str:
    material = json.dumps([VERSION, plan.protocol_sha256, paper_text.source_sha256,
                           extraction_model, judge_model, extraction_template, judge_template, facet.id, top_k,
                           spec.instruction, spec.query],
                          ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_fidelity_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "configs" / "prompts" / FIDELITY_PROMPT_NAME
    return path.read_text(encoding="utf-8")


def _status_from_card_field(field) -> str:
    return {
        "not_reported": "not_reported",
        "rejected": "rejected",
        "llm_error": "llm_error",
        "invalid_format": "llm_error",
        "no_context": "missing",
    }.get(field.status, "missing")


def _record_from_reused(paper_id, facet, card_field, fidelity_report, queue) -> tuple[FacetEvidenceRecord, list]:
    review_items = []
    if card_field.status != "verified":
        status = _status_from_card_field(card_field)
        return FacetEvidenceRecord(
            paper_id=paper_id, facet_id=facet.id, source_facet_ids=facet.source_facet_ids,
            rq_ids=facet.rq_ids, status=status, source_field=facet.reuse_field,
            value=card_field.value, quote=card_field.quote, chunk_id=card_field.chunk_id,
            page=card_field.page, reused=True, note=f"Champ source : {card_field.status}",
        ), review_items

    if fidelity_report is None:
        status, note = "llm_error", "Résultat du juge de fidélité absent pour le champ source."
    else:
        fidelity_report = _apply_human_decision(fidelity_report, queue)
        status = {
            "verified": "found", "human_review": "human_review", "rejected": "rejected",
            "llm_error": "llm_error", "skipped": "missing",
        }[fidelity_report.status]
        note = fidelity_report.reason
        if status == "human_review":
            item = queue.open_item("G5", "evidence", f"{paper_id}:{facet.id}",
                                   f"Revue humaine de la facette {facet.label} : {note or 'jugement partiel'}")
            if item.status == "open":
                review_items.append(item)

    return FacetEvidenceRecord(
        paper_id=paper_id, facet_id=facet.id, source_facet_ids=facet.source_facet_ids,
        rq_ids=facet.rq_ids, status=status, source_field=facet.reuse_field,
        value=card_field.value, quote=card_field.quote, chunk_id=card_field.chunk_id,
        page=card_field.page, reused=True, note=note, fidelity=fidelity_report,
        input_fingerprint=(fidelity_report.input_fingerprint if fidelity_report else ""),
    ), review_items


def _record_from_new(paper_id, facet, field, judge_model, judge_client, judge_template, chunks, fingerprint, queue,
                     review_suffix=""):
    common = dict(paper_id=paper_id, facet_id=facet.id, source_facet_ids=facet.source_facet_ids,
                  rq_ids=facet.rq_ids, source_field=facet.id, value=field.value, quote=field.quote,
                  chunk_id=field.chunk_id, page=field.page, reused=False, input_fingerprint=fingerprint)
    if field.status != "verified":
        status = _status_from_card_field(field)
        return FacetEvidenceRecord(**common, status=status, note=field.note), None
    chunk = next((chunk for chunk in chunks if chunk.chunk_id == field.chunk_id), None)
    report = _report(paper_id, field, chunk, judge_model, judge_client, judge_template, fingerprint)
    status = {
        "verified": "found", "human_review": "human_review", "rejected": "rejected",
        "llm_error": "llm_error", "skipped": "missing",
    }[report.status]
    item = None
    if report.status == "human_review":
        target_id = f"{paper_id}:{facet.id}" + (f":{review_suffix}" if review_suffix else "")
        item = queue.open_item("G5", "evidence", target_id,
                              f"Revue humaine de la facette {facet.label} : {report.reason}")
        if item.status != "open":
            item = None
    return FacetEvidenceRecord(**common, status=status, note=report.reason, fidelity=report), item


def _keep_prior_after_repair_error(previous: FacetEvidenceRecord, attempted: FacetEvidenceRecord) -> FacetEvidenceRecord:
    """Do not replace a reviewed status with a transient provider failure."""
    if previous is None:
        return attempted.model_copy(update={"repair_attempted": False})
    return previous.model_copy(update={"repair_attempted": False})


class FacetEvidenceStage:
    name = ARTIFACT

    def __init__(self, top_k: int = 3, max_papers: int | None = None, pause: float = 2.1,
                 force: bool = False, repair_missing: bool = False, sleep=time.sleep, clock=time.monotonic):
        if top_k <= 0:
            raise ValueError("top_k doit être > 0")
        if max_papers is not None and max_papers <= 0:
            raise ValueError("max_papers doit être > 0")
        if pause < 0:
            raise ValueError("pause doit être >= 0")
        self.top_k, self.max_papers, self.pause, self.force = top_k, max_papers, pause, force
        self.repair_missing = repair_missing
        self._sleep, self._clock = sleep, clock

    def run(self, ctx) -> StageResult:
        if not ctx.settings.judge_is_independent:
            return StageResult(status="failed", message="Le modèle juge doit différer du modèle d'extraction.")
        try:
            plan = ctx.store.read_artifact("facet_plan", FacetPlanOutcome)
            cards_index = ctx.store.read_artifact("study_cards", StudyCardsOutcome)
            extraction = ctx.store.read_artifact("extraction", ExtractionOutcome)
            fidelity = ctx.store.read_artifact("fidelity", FidelityOutcome)
        except ArtifactNotFound as exc:
            return StageResult(status="failed", message=f"artefact requis introuvable : {exc}")
        gate = ctx.queue.get(plan.gate_id)
        if gate is None or gate.status != "closed":
            item = gate or ctx.queue.open_item("G1", "facet_plan", plan.gate_id.split(":", 2)[2],
                                               "Approuver le plan normalisé des facettes avant l'extraction 6b.")
            return StageResult(status="waiting_human", artifact="facet_plan", review_items=[item],
                               message="L'extraction 6b attend l'approbation G1.")
        if gate.decision != "approved":
            return StageResult(status="failed", artifact="facet_plan", message=f"Plan de facettes refusé en G1 : {gate.decision}")
        papers = [p for p in cards_index.papers if p.status != "failed"]
        if self.max_papers is not None and len(papers) > self.max_papers:
            return StageResult(status="failed", message=f"{len(papers)} papiers dépassent --max-papers {self.max_papers}; aucun appel fait")

        extraction_files = {p.paper_id: p.file for p in extraction.papers if p.status == "extracted"}
        card_entries = {p.paper_id: p for p in cards_index.papers}
        fidelity_by_key = {(r.paper_id, r.field_id): r for p in fidelity.papers for r in p.reports}
        previous = {}
        if not self.force and ctx.store.exists(ARTIFACT):
            try:
                old = ctx.store.read_artifact(ARTIFACT, FacetEvidenceOutcome)
                if old.plan_hash == plan.protocol_sha256 and old.extraction_model == ctx.settings.model_for("extraction") and old.judge_model == ctx.settings.model_for("judge"):
                    previous = {(r.paper_id, r.facet_id): r for p in old.papers for r in p.evidence}
            except Exception:
                previous = {}

        extraction_model = ctx.settings.model_for("extraction")
        judge_model = ctx.settings.model_for("judge")
        extraction_template = load_prompt()
        judge_template = load_fidelity_prompt()
        pacer = _CallPacer(self.pause, self._sleep, self._clock)
        extraction_client = _PacedClient(ctx.llm("extraction"), pacer)
        judge_client = _PacedClient(ctx.llm("judge"), pacer)
        cached_calls = 0
        result_papers, review_items = [], []
        stop_reason = ""
        consecutive_errors = 0
        repair_errors = 0

        for entry in papers:
            if entry.paper_id not in extraction_files:
                continue
            card_entry = card_entries[entry.paper_id]
            card = StudyCard.model_validate_json((ctx.store.run_dir / card_entry.file).read_text(encoding="utf-8"))
            text = load_paper_text(ctx.store.run_dir / extraction_files[entry.paper_id])
            if text is None:
                continue
            card_fields = {field.field_id: field for field in card.fields}
            chunks = text.chunks
            records = []
            for facet in plan.facets:
                spec = _SUPPLEMENTAL_FIELDS.get(facet.id) or FieldSpec(
                    facet.id, facet.label, facet.description, facet.query_en)
                fingerprint = _fingerprint(plan, text, extraction_model, judge_model,
                                           extraction_template, judge_template, facet, spec, self.top_k)
                cached = previous.get((entry.paper_id, facet.id))

                if facet.reuse_field:
                    # A previously repaired result is resumable and retains its separate G5 item.
                    if (self.repair_missing and cached is not None and cached.repair_attempted
                            and cached.input_fingerprint == fingerprint):
                        record = cached
                        if cached.fidelity is not None:
                            report = _apply_human_decision(cached.fidelity, ctx.queue,
                                item_id=f"G5:evidence:{entry.paper_id}:{facet.id}:repair")
                            status = {"verified": "found", "human_review": "human_review", "rejected": "rejected",
                                      "llm_error": "llm_error", "skipped": "missing"}[report.status]
                            record = cached.model_copy(update={"fidelity": report, "status": status, "note": report.reason})
                        records.append(record)
                        if record.status == "human_review":
                            item = ctx.queue.open_item("G5", "evidence", f"{entry.paper_id}:{facet.id}:repair",
                                f"Revue humaine de la facette {facet.label} après nouvelle extraction : {record.note}")
                            if item.status == "open": review_items.append(item)
                        continue

                    source = card_fields.get(facet.reuse_field)
                    if source is None:
                        record = FacetEvidenceRecord(paper_id=entry.paper_id, facet_id=facet.id,
                            source_facet_ids=facet.source_facet_ids, rq_ids=facet.rq_ids, status="missing",
                            source_field=facet.reuse_field, reused=True, note="Champ source absent de la fiche.")
                    else:
                        record, items = _record_from_reused(entry.paper_id, facet, source,
                            fidelity_by_key.get((entry.paper_id, facet.reuse_field)), ctx.queue)
                        review_items.extend(items)
                    if self.repair_missing and record.status not in ("found", "not_reported"):
                        previous_record = record
                        if stop_reason:
                            records.append(record)
                            continue
                        try:
                            field = fill_field(extraction_client, spec, chunks, extraction_template, self.top_k)
                            record, item = _record_from_new(entry.paper_id, facet, field, judge_model, judge_client,
                                judge_template, chunks, fingerprint, ctx.queue, review_suffix="repair")
                            record = record.model_copy(update={"repair_attempted": True})
                            if item is not None: review_items.append(item)
                        except BudgetExceeded as exc:
                            stop_reason = f"budget atteint : {exc}"
                            record = FacetEvidenceRecord(paper_id=entry.paper_id, facet_id=facet.id,
                                source_facet_ids=facet.source_facet_ids, rq_ids=facet.rq_ids, status="llm_error",
                                source_field=facet.id, input_fingerprint=fingerprint, repair_attempted=True, note=stop_reason)
                        if record.status == "llm_error":
                            repair_errors += 1
                            consecutive_errors += 1
                            error_note = record.note or "Erreur de réparation LLM."
                            if consecutive_errors >= 3:
                                stop_reason = error_note
                            ctx.trace.log("facet_evidence", "repair_failed", paper_id=entry.paper_id,
                                          facet=facet.id, reason=error_note)
                            record = _keep_prior_after_repair_error(previous_record, record)
                        else:
                            consecutive_errors = 0
                    records.append(record)
                    continue

                should_repair_cached = (self.repair_missing and cached is not None
                    and cached.status not in ("found", "not_reported") and not cached.repair_attempted)
                if (cached is not None and cached.input_fingerprint == fingerprint and cached.status != "llm_error"
                        and not should_repair_cached):
                    if cached.fidelity is not None:
                        item_id = f"G5:evidence:{entry.paper_id}:{facet.id}:repair" if cached.repair_attempted else None
                        report = _apply_human_decision(cached.fidelity, ctx.queue, item_id=item_id)
                        status = {"verified": "found", "human_review": "human_review", "rejected": "rejected",
                                  "llm_error": "llm_error", "skipped": "missing"}[report.status]
                        cached = cached.model_copy(update={"fidelity": report, "status": status, "note": report.reason})
                    records.append(cached)
                    cached_calls += 1
                    if cached.status == "human_review":
                        item = ctx.queue.open_item("G5", "evidence", f"{entry.paper_id}:{facet.id}",
                                                   f"Revue humaine de la facette {facet.label} : {cached.note}")
                        if item.status == "open": review_items.append(item)
                    continue

                if stop_reason:
                    if self.repair_missing and cached is not None:
                        records.append(cached)
                        continue
                    records.append(FacetEvidenceRecord(paper_id=entry.paper_id, facet_id=facet.id,
                        source_facet_ids=facet.source_facet_ids, rq_ids=facet.rq_ids, status="llm_error",
                        source_field=facet.id, input_fingerprint=fingerprint, note=stop_reason))
                    continue

                try:
                    field = fill_field(extraction_client, spec, chunks, extraction_template, self.top_k)
                    record, item = _record_from_new(entry.paper_id, facet, field, judge_model, judge_client,
                                                    judge_template, chunks, fingerprint, ctx.queue,
                                                    review_suffix="repair" if self.repair_missing else "")
                    if self.repair_missing:
                        record = record.model_copy(update={"repair_attempted": True})
                except BudgetExceeded as exc:
                    stop_reason = f"budget atteint : {exc}"
                    record = FacetEvidenceRecord(paper_id=entry.paper_id, facet_id=facet.id,
                        source_facet_ids=facet.source_facet_ids, rq_ids=facet.rq_ids, status="llm_error",
                        source_field=facet.id, input_fingerprint=fingerprint, repair_attempted=self.repair_missing,
                        note=stop_reason)
                    item = None
                if self.repair_missing and record.status == "llm_error":
                    repair_errors += 1
                    consecutive_errors += 1
                    error_note = record.note or "Erreur de réparation LLM."
                    if consecutive_errors >= 3:
                        stop_reason = error_note
                    ctx.trace.log("facet_evidence", "repair_failed", paper_id=entry.paper_id,
                                  facet=facet.id, reason=error_note)
                    if cached is not None:
                        record = _keep_prior_after_repair_error(cached, record)
                elif record.status != "llm_error":
                    consecutive_errors = 0
                records.append(record)
                if item is not None:
                    review_items.append(item)
                if record.status == "llm_error":
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        stop_reason = record.note or "Trois erreurs LLM consécutives; arrêt précoce pour économiser le budget."
                else:
                    consecutive_errors = 0
                ctx.trace.log("facet_evidence", "facet_done", paper_id=entry.paper_id,
                              facet=facet.id, status=record.status, reused=False)

            result_papers.append(FacetEvidencePaper(paper_id=entry.paper_id, evidence=records))
            outcome = self._outcome(plan, extraction_model, judge_model, result_papers, extraction_client, judge_client)
            ctx.store.write_artifact(ARTIFACT, FacetEvidenceOutcome, outcome)

        if not result_papers:
            return StageResult(status="failed", message="Aucun papier extrait n'est disponible pour les facettes.")
        outcome = self._outcome(plan, extraction_model, judge_model, result_papers, extraction_client, judge_client)
        ctx.store.write_artifact(ARTIFACT, FacetEvidenceOutcome, outcome)
        ctx.trace.log("facet_evidence", "written", **outcome.summary.model_dump())
        if repair_errors:
            return StageResult(status="failed", artifact=ARTIFACT,
                               message=f"Réparation interrompue : {repair_errors} appels sans réponse fiable; preuves précédentes conservées; relance possible.")
        if review_items:
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=review_items,
                               message=f"{len(review_items)} éléments de preuve attendent une revue G5")
        if outcome.summary.llm_error:
            return StageResult(status="failed", artifact=ARTIFACT,
                               message=f"Étape 6b interrompue : {outcome.summary.llm_error} éléments en erreur, reprise possible sans refaire les résultats conservés.")
        return StageResult(status="done", artifact=ARTIFACT,
                           message=f"Éléments réutilisés depuis 6a : {outcome.summary.reused}; appels déjà en cache : {cached_calls}.")

    @staticmethod
    def _outcome(plan, extraction_model, judge_model, papers, extraction_client, judge_client):
        rows = [r for p in papers for r in p.evidence]
        counts = {status: sum(r.status == status for r in rows)
                  for status in ("found", "not_reported", "missing", "human_review", "rejected", "llm_error")}
        return FacetEvidenceOutcome(version=VERSION, plan_hash=plan.protocol_sha256,
            judge_model=judge_model, extraction_model=extraction_model, papers=papers,
            summary=FacetEvidenceSummary(papers=len(papers), **counts,
                reused=sum(r.reused for r in rows), llm_calls=extraction_client.client.calls + judge_client.client.calls,
                tokens=extraction_client.client.total_tokens + judge_client.client.total_tokens))
