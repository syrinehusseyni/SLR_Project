"""Step 8: evidence-grounded, per-study quality reporting appraisal."""
import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded, LLMError
from slrresearch.core.schemas.extraction import ExtractionOutcome
from slrresearch.core.schemas.facet_evidence import FacetEvidenceOutcome
from slrresearch.core.schemas.facet_matrix import FacetMatrixOutcome
from slrresearch.core.schemas.facet_plan import FacetPlanOutcome
from slrresearch.core.schemas.quality import (
    QualityAssessment, QualityOutcome, QualityPaper, QualitySummary,
)
from slrresearch.evidence.fidelity import ROLE as JUDGE_ROLE
from slrresearch.extraction.reader import load_paper_text
from slrresearch.extraction.study_card import flat
from slrresearch.quality.checklist import (
    CHECKLIST, PROMPT_NAME, RUBRIC_VERSION, build_prompt, fingerprint, parse_response,
)

ARTIFACT = "quality"
VERSION = "quality-appraisal-v1"
REVIEW_CONFIDENCE = 0.75


def load_prompt(prompt_dir: Path | None = None) -> str:
    path = prompt_dir / PROMPT_NAME if prompt_dir else Path("configs/prompts") / PROMPT_NAME
    return path.read_text(encoding="utf-8")


def _evidence_context(paper) -> list[dict]:
    by_category = {row.facet_id: row for row in paper.evidence}
    context = []
    for criterion in CHECKLIST:
        rows = []
        for category in criterion.source_categories:
            row = by_category.get(category)
            if row is None:
                rows.append({"category": category, "status": "missing_from_step_6"})
            elif row.status == "found":
                rows.append({"category": category, "status": "found", "value": row.value,
                             "quote": row.quote, "chunk_id": row.chunk_id, "page": row.page})
            else:
                # Never pass rejected or unverified claims to the quality judge as facts.
                rows.append({"category": category, "status": row.status,
                             "note": "No accepted excerpt supplied for this category."})
        context.append({"criterion_id": criterion.id, "label": criterion.label,
                        "question": criterion.question, "evidence": rows})
    return context


def _gate_id(paper_id: str, fingerprint_value: str, assessments: list[QualityAssessment]) -> str:
    rows = [{"criterion_id": row.criterion_id, "rating": row.rating, "confidence": row.confidence,
             "quote": row.quote, "chunk_id": row.chunk_id, "status": row.status} for row in assessments]
    digest = hashlib.sha256(json.dumps([fingerprint_value, rows], sort_keys=True,
                                        ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return f"G4:quality:{paper_id}:{digest}"


def _paper_status(assessments: list[QualityAssessment]) -> str:
    statuses = {item.status for item in assessments}
    if "llm_error" in statuses:
        return "incomplete"
    if "human_review" in statuses:
        return "waiting_human"
    if "rejected" in statuses:
        return "rejected"
    return "complete"


def _apply_gate(paper: QualityPaper, queue) -> QualityPaper:
    pending = [item for item in paper.assessments if item.status == "human_review"]
    if not pending or not paper.review_item_id:
        return paper.model_copy(update={"status": _paper_status(paper.assessments)})
    item = queue.get(paper.review_item_id)
    if item is None:
        return paper
    if item.status == "open":
        return paper.model_copy(update={"status": "waiting_human"})
    updated = []
    for assessment in paper.assessments:
        if assessment.status != "human_review":
            updated.append(assessment)
            continue
        status = "accepted" if item.decision == "approved" else "rejected"
        updated.append(assessment.model_copy(update={"status": status,
            "review_decision": item.decision if item.decision in ("approved", "rejected") else None,
            "reviewer": item.decided_by,
            "note": (assessment.note + f" Décision G4 : {item.decision} par {item.decided_by}.").strip()}))
    return paper.model_copy(update={"assessments": updated, "status": _paper_status(updated)})


class QualityStage:
    name = ARTIFACT

    def __init__(self, max_papers: int | None = None, pause: float = 2.1, force: bool = False,
                 review_confidence: float = REVIEW_CONFIDENCE, prompt_dir: Path | None = None,
                 progress=None, sleep=time.sleep, clock=time.monotonic):
        if max_papers is not None and max_papers <= 0:
            raise ValueError("max_papers doit être > 0")
        if pause < 0:
            raise ValueError("pause doit être >= 0")
        if not 0 <= review_confidence <= 1:
            raise ValueError("review_confidence doit être entre 0 et 1")
        self.max_papers, self.pause, self.force = max_papers, pause, force
        self.review_confidence = review_confidence
        self.prompt_dir, self.progress, self._sleep, self._clock = prompt_dir, progress, sleep, clock

    @staticmethod
    def _outcome(papers, model, plan_hash, client) -> QualityOutcome:
        assessments = [item for paper in papers for item in paper.assessments]
        ratings = {rating: sum(item.rating == rating for item in assessments)
                   for rating in ("yes", "partial", "no", "not_reported")}
        counts = {status: sum(item.status == status for item in assessments)
                  for status in ("accepted", "human_review", "rejected", "llm_error")}
        return QualityOutcome(
            version=VERSION, rubric_version=RUBRIC_VERSION, judge_model=model,
            plan_hash=plan_hash, papers=papers,
            summary=QualitySummary(
                papers=len(papers), complete=sum(p.status == "complete" for p in papers),
                waiting_human=sum(p.status == "waiting_human" for p in papers),
                incomplete=sum(p.status == "incomplete" for p in papers),
                rejected=sum(p.status == "rejected" for p in papers),
                assessments=len(assessments), accepted=counts["accepted"],
                human_review=counts["human_review"], rejected_items=counts["rejected"],
                llm_errors=counts["llm_error"], ratings=ratings,
                llm_calls=client.calls if client else 0, tokens=client.total_tokens if client else 0,
            ),
        )

    def run(self, ctx) -> StageResult:
        if not ctx.settings.judge_is_independent:
            return StageResult(status="failed", message="Le juge qualité doit différer du modèle d'extraction.")
        try:
            plan = ctx.store.read_artifact("facet_plan", FacetPlanOutcome)
            evidence = ctx.store.read_artifact("facet_evidence", FacetEvidenceOutcome)
            matrix = ctx.store.read_artifact("facet_matrix", FacetMatrixOutcome)
            extraction = ctx.store.read_artifact("extraction", ExtractionOutcome)
        except ArtifactNotFound as exc:
            return StageResult(status="failed", message=f"artefact requis introuvable : {exc}")
        if evidence.plan_hash != plan.protocol_sha256 or matrix.plan_hash != plan.protocol_sha256:
            return StageResult(status="failed", message="les artefacts de facettes ne correspondent pas au même plan")
        if matrix.summary.missing or matrix.summary.blocking_missing:
            return StageResult(status="failed", message="la matrice comporte encore des facettes manquantes; finir l'étape 7 avant la qualité")
        entries = {entry.paper_id: entry for entry in extraction.papers if entry.status == "extracted"}
        evidence_papers = [paper for paper in evidence.papers if paper.paper_id in entries]
        if self.max_papers is not None and len(evidence_papers) > self.max_papers:
            return StageResult(status="failed", message=f"{len(evidence_papers)} articles dépassent --max-papers {self.max_papers}; aucun appel fait")
        if not evidence_papers:
            return StageResult(status="failed", message="aucun article extrait n'est disponible pour l'évaluation qualité")
        try:
            prompt_template = load_prompt(self.prompt_dir)
        except OSError as exc:
            return StageResult(status="failed", message=f"prompt qualité introuvable : {exc}")

        model_name = ctx.settings.model_for(JUDGE_ROLE)
        prior = {}
        if not self.force and ctx.store.exists(ARTIFACT):
            try:
                previous = ctx.store.read_artifact(ARTIFACT, QualityOutcome)
                if previous.rubric_version == RUBRIC_VERSION and previous.judge_model == model_name and previous.plan_hash == plan.protocol_sha256:
                    prior = {paper.paper_id: paper for paper in previous.papers}
            except Exception:
                prior = {}

        client = None
        results = []
        review_items = []
        last_call = None
        total = len(evidence_papers)
        for index, evidence_paper in enumerate(evidence_papers, start=1):
            entry = entries[evidence_paper.paper_id]
            paper_text = load_paper_text(ctx.store.run_dir / entry.file)
            if paper_text is None:
                fingerprint_value = hashlib.sha256(f"{evidence_paper.paper_id}:missing-text:{RUBRIC_VERSION}".encode()).hexdigest()
                assessments = [QualityAssessment(criterion_id=c.id, status="llm_error", note="Texte extrait introuvable ou invalide.") for c in CHECKLIST]
                paper = QualityPaper(paper_id=evidence_paper.paper_id, fingerprint=fingerprint_value,
                                     status="incomplete", assessments=assessments, note="Texte extrait introuvable.")
                results.append(paper)
                continue
            evidence_context = _evidence_context(evidence_paper)
            context = {"paper_id": evidence_paper.paper_id,
                       "scope_note": "Assess reporting only from accepted Step 6 excerpts shown here; this is not a full-text risk-of-bias review.",
                       "criteria": evidence_context}
            fingerprint_value = fingerprint(evidence_paper.paper_id, paper_text.source_sha256,
                                            evidence_context, model_name, prompt_template,
                                            self.review_confidence)
            cached = prior.get(evidence_paper.paper_id)
            if cached is not None and cached.fingerprint == fingerprint_value and cached.status != "incomplete":
                paper = _apply_gate(cached, ctx.queue)
                if paper.status == "waiting_human" and paper.review_item_id:
                    item = ctx.queue.open_item("G4", "quality", paper.review_item_id.removeprefix("G4:quality:"),
                                               paper.note or "Revue humaine de l'évaluation qualité.")
                    if item.status == "open":
                        review_items.append(item)
                results.append(paper)
                if self.progress:
                    self.progress(index, total, paper.paper_id, paper.status)
                continue

            if client is None:
                client = ctx.llm(JUDGE_ROLE)
            if last_call is not None and self.pause:
                remaining = self.pause - (self._clock() - last_call)
                if remaining > 0:
                    self._sleep(remaining)
            try:
                response = client.call(build_prompt(prompt_template, context))
                last_call = self._clock()
                proposed = parse_response(response.text)
                chunks = {chunk.chunk_id: chunk for chunk in paper_text.chunks}
                assessments = []
                for answer in proposed.assessments:
                    chunk = chunks.get(answer.chunk_id)
                    quote_ok = bool(answer.rating == "not_reported" or
                                    (chunk and chunk.paper_id == evidence_paper.paper_id
                                     and flat(answer.quote) in flat(chunk.text)))
                    if not quote_ok:
                        assessments.append(QualityAssessment(
                            criterion_id=answer.criterion_id, rating=answer.rating,
                            confidence=answer.confidence, rationale=answer.rationale,
                            quote=answer.quote, chunk_id=answer.chunk_id,
                            page=chunk.page if chunk else None, quote_in_chunk=False,
                            status="llm_error", note="Citation absente du chunk source ou chunk_id incorrect; évaluation non acceptée."))
                        continue
                    needs_review = (answer.rating in ("partial", "no", "not_reported")
                                    or answer.confidence < self.review_confidence)
                    assessments.append(QualityAssessment(
                        criterion_id=answer.criterion_id, rating=answer.rating,
                        confidence=answer.confidence, rationale=answer.rationale,
                        quote=answer.quote, chunk_id=answer.chunk_id,
                        page=chunk.page if chunk else None, quote_in_chunk=answer.rating != "not_reported",
                        status="human_review" if needs_review else "accepted",
                        note=("À revoir : rating partiel/négatif, confiance basse ou information non trouvée dans les extraits."
                              if needs_review else "Citation retrouvée dans le chunk source.")))
                paper = QualityPaper(paper_id=evidence_paper.paper_id, fingerprint=fingerprint_value,
                                     status="incomplete", assessments=assessments)
                pending = [item for item in assessments if item.status == "human_review"]
                if pending:
                    gate_id = _gate_id(paper.paper_id, fingerprint_value, assessments)
                    reason = ("Revoir les critères qualité " + ", ".join(item.criterion_id for item in pending)
                              + ". Vérifier les citations dans le PDF; pour not_reported, vérifier le texte complet avant de confirmer l'absence.")
                    target_id = gate_id.removeprefix("G4:quality:")
                    queue_item = ctx.queue.open_item("G4", "quality", target_id, reason)
                    paper = paper.model_copy(update={"review_item_id": gate_id, "note": reason})
                    if queue_item.status == "open":
                        paper = paper.model_copy(update={"status": "waiting_human"})
                        review_items.append(queue_item)
                    else:
                        paper = _apply_gate(paper, ctx.queue)
                else:
                    paper = paper.model_copy(update={"status": _paper_status(assessments)})
            except BudgetExceeded as exc:
                last_call = self._clock()
                assessments = [QualityAssessment(criterion_id=c.id, status="llm_error", note=str(exc)) for c in CHECKLIST]
                paper = QualityPaper(paper_id=evidence_paper.paper_id, fingerprint=fingerprint_value,
                                     status="incomplete", assessments=assessments, note=str(exc))
            except (LLMError, ValueError, ValidationError) as exc:
                last_call = self._clock()
                assessments = [QualityAssessment(criterion_id=c.id, status="llm_error", note=str(exc)) for c in CHECKLIST]
                paper = QualityPaper(paper_id=evidence_paper.paper_id, fingerprint=fingerprint_value,
                                     status="incomplete", assessments=assessments, note=str(exc))

            results.append(paper)
            outcome = self._outcome(results, model_name, plan.protocol_sha256, client)
            ctx.store.write_artifact(ARTIFACT, QualityOutcome, outcome)
            ctx.trace.log("quality", "paper_appraised", paper_id=paper.paper_id,
                          status=paper.status, ratings={item.criterion_id: item.rating for item in paper.assessments},
                          review_item=paper.review_item_id)
            if self.progress:
                self.progress(index, total, paper.paper_id, paper.status)

        outcome = self._outcome(results, model_name, plan.protocol_sha256, client)
        ctx.store.write_artifact(ARTIFACT, QualityOutcome, outcome)
        ctx.trace.log("quality", "written", **outcome.summary.model_dump())
        if outcome.summary.incomplete:
            return StageResult(status="failed", artifact=ARTIFACT,
                               message=f"{outcome.summary.incomplete} fiches qualité incomplètes; relancez pour reprendre les erreurs.")
        if review_items or outcome.summary.waiting_human:
            open_items = [item for paper in results if paper.status == "waiting_human"
                          for item in [ctx.queue.get(paper.review_item_id)] if item and item.status == "open"]
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=open_items,
                               message=f"{len(open_items)} articles attendent une revue qualité G4")
        return StageResult(status="done", artifact=ARTIFACT,
                           message="Évaluations qualité terminées; aucune revue G4 ne reste ouverte.")
