"""Étape 5b : une fiche d'étude par papier extrait (design, données, modèles, résultats, limites), citations vérifiées.

PLAN (papiers `extracted` de l'artefact `extraction`) -> EXECUTE (un appel LLM par champ, rôle `extraction`)
-> OBSERVE (statut de chaque champ) -> DECIDE : fiche `complete` ou `incomplete` (champs en panne LLM à relancer).

Reprise : un champ déjà établi (même empreinte : chunks, prompt, modèle, réglages) n'est jamais redemandé ; seuls les champs
en `llm_error` sont retentés. Une fiche est écrite dès que son papier est fini. Le budget d'appels est contrôlé avant chaque
papier : si le papier suivant ne tient pas dans ce qui reste, on s'arrête proprement (les fiches faites sont gardées).
Pas de porte humaine : un champ rejeté ou sans contexte est signalé, il ne bloque rien.
"""
import time
from pathlib import Path

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded
from slrresearch.core.schemas.extraction import Chunk, ExtractedPaperEntry, ExtractionOutcome
from slrresearch.core.schemas.study_card import (
    RETRY_STATUSES, CardEntry, CardField, StudyCard, StudyCardsLimits, StudyCardsOutcome, StudyCardsSummary,
)
from slrresearch.extraction.reader import load_paper_text
from slrresearch.extraction.study_card import (
    FIELDS, PROMPT_NAME, card_file, fill_field, fingerprint, load_card, load_prompt, save_card,
)

ARTIFACT = "study_cards"
ROLE = "extraction"
DEFAULT_TOP_K = 3


def count_by_status(fields: list[CardField]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in fields:
        counts[f.status] = counts.get(f.status, 0) + 1
    return counts


def entry_from_card(card: StudyCard, run_dir: Path, path: Path) -> CardEntry:
    incomplete = any(f.status in RETRY_STATUSES for f in card.fields)
    return CardEntry(paper_id=card.paper_id, status="incomplete" if incomplete else "complete",
                     file=path.relative_to(run_dir).as_posix(), counts=count_by_status(card.fields))


class StudyCardStage:
    name = "study_cards"

    def __init__(self, top_k: int = DEFAULT_TOP_K, max_papers: int | None = None, pause: float = 0.0, force: bool = False,
                 progress=None, sleep=time.sleep, prompt_dir: Path | None = None):
        if top_k <= 0:
            raise ValueError("top_k doit être > 0")
        if pause < 0:
            raise ValueError("pause doit être >= 0")
        self.top_k = top_k
        self.max_papers = max_papers  # refuse de démarrer si plus de N papiers doivent passer par le LLM
        self.pause = pause            # secondes entre deux appels LLM (ex. 3 pour Groq)
        self.force = force            # ignorer les fiches déjà faites
        self.progress = progress      # fonction (index, total, paper_id, statut) ; None = silencieux
        self._sleep = sleep
        self.prompt_dir = prompt_dir

    def run(self, ctx) -> StageResult:
        try:
            index = ctx.store.read_artifact("extraction", ExtractionOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « extraction » introuvable : lancez d'abord l'étape extraction.")
        papers = [p for p in index.papers if p.status == "extracted"]
        skipped = len(index.papers) - len(papers)
        if not papers:
            return StageResult(status="failed", message="aucun papier extrait : rien à résumer.")
        template = load_prompt(directory=self.prompt_dir)
        model = ctx.settings.model_for(ROLE)
        run_dir = ctx.store.run_dir

        entries: dict[str, CardEntry] = {}
        reused = 0
        work: list[tuple[ExtractedPaperEntry, list[Chunk], str, dict[str, CardField], list]] = []
        for paper in papers:
            text = load_paper_text(run_dir / paper.file)
            if text is None:
                entries[paper.paper_id] = CardEntry(paper_id=paper.paper_id, status="failed",
                                                    reason="fichier d'extraction illisible : relancez l'étape extraction")
                continue
            fp = fingerprint(template, self.top_k, model, text.chunks)
            kept: dict[str, CardField] = {}
            old = None if self.force else load_card(card_file(run_dir, paper.paper_id))
            if old is not None and old.fingerprint == fp:
                kept = {f.field_id: f for f in old.fields if f.status not in RETRY_STATUSES}
            pending = [spec for spec in FIELDS if spec.id not in kept]
            if pending:
                work.append((paper, text.chunks, fp, kept, pending))
            else:
                entries[paper.paper_id] = entry_from_card(old, run_dir, card_file(run_dir, paper.paper_id))
                reused += 1

        ctx.trace.log("study_card", "started", extracted=len(papers), reused=reused, to_process=len(work), model=model)
        if self.max_papers is not None and len(work) > self.max_papers:
            ctx.trace.log("study_card", "refused_too_many", to_process=len(work), max_papers=self.max_papers)
            return StageResult(status="failed", message=(
                f"{len(work)} papiers doivent passer par le LLM, plus que la limite --max-papers {self.max_papers}. "
                "Aucun appel n'a été fait. Augmentez --max-papers."))

        client = ctx.llm(ROLE) if work else None
        budget_stop = ""
        calls_made = 0
        try:
            for position, (paper, chunks, fp, kept, pending) in enumerate(work, 1):
                if ctx.budget is not None and ctx.budget.remaining_calls() < len(pending):
                    budget_stop = f"budget d'appels insuffisant pour le papier {paper.paper_id} ({len(pending)} appels nécessaires, {ctx.budget.remaining_calls()} restants)"
                    break
                fields = dict(kept)
                try:
                    for spec in pending:
                        if calls_made and self.pause:
                            self._sleep(self.pause)
                        calls_made += 1
                        field = fill_field(client, spec, chunks, template, self.top_k)
                        fields[spec.id] = field
                        ctx.trace.log("study_card", "field_done", paper_id=paper.paper_id, field=spec.id, status=field.status)
                except BudgetExceeded as exc:
                    budget_stop = f"{exc} (papier {paper.paper_id} interrompu : ses champs seront redemandés à la reprise)"
                    break
                card = StudyCard(paper_id=paper.paper_id, fingerprint=fp, model=model,
                                 fields=[fields[spec.id] for spec in FIELDS])
                path = card_file(run_dir, paper.paper_id)
                save_card(path, card)
                entries[paper.paper_id] = entry_from_card(card, run_dir, path)
                counts = entries[paper.paper_id].counts
                ctx.trace.log("study_card", "paper_done", paper_id=paper.paper_id, **counts)
                self._tell(position, len(work), paper.paper_id, ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
        except KeyboardInterrupt:
            ctx.trace.log("study_card", "interrupted", done=len(entries), total=len(papers))
            raise  # les fiches déjà écrites sont conservées : la reprise les réutilise

        if budget_stop:
            ctx.trace.log("study_card", "budget_stop", reason=budget_stop)
            done = sum(1 for e in entries.values() if e.status != "failed")
            return StageResult(status="failed", message=(
                f"arrêt : {budget_stop}. {done} fiches sur {len(papers)} sont faites et conservées ; relancez la même commande "
                "avec un --max-calls plus grand pour reprendre."))
        return self._finalize(ctx, model, papers, entries, skipped, client)

    def _tell(self, index: int, total: int, paper_id: str, status: str) -> None:
        if self.progress is not None:
            self.progress(index, total, paper_id, status)

    def _finalize(self, ctx, model: str, papers: list[ExtractedPaperEntry], entries: dict[str, CardEntry], skipped: int, client) -> StageResult:
        ordered = [entries[p.paper_id] for p in papers]
        fields_total: dict[str, int] = {}
        for entry in ordered:
            for status, n in entry.counts.items():
                fields_total[status] = fields_total.get(status, 0) + n
        summary = StudyCardsSummary(
            total=len(ordered), complete=sum(e.status == "complete" for e in ordered),
            incomplete=sum(e.status == "incomplete" for e in ordered), failed=sum(e.status == "failed" for e in ordered),
            skipped_not_extracted=skipped, llm_calls=client.calls if client else 0,
            tokens=client.total_tokens if client else 0, fields=fields_total)
        limits = StudyCardsLimits(model=model, prompt=PROMPT_NAME, top_k=self.top_k, pause=self.pause,
                                  max_papers=self.max_papers, fields=[f.id for f in FIELDS])
        ctx.store.write_artifact(ARTIFACT, StudyCardsOutcome, StudyCardsOutcome(limits=limits, papers=ordered, summary=summary).model_dump())
        ctx.trace.log("study_card", "written", **summary.model_dump(exclude={"fields"}))
        if summary.complete == 0:
            bad = [f"{e.paper_id} ({e.reason or 'champs en panne LLM'})" for e in ordered if e.status != "complete"]
            return StageResult(status="failed", message="aucune fiche complète : " + " ; ".join(bad[:5]))
        note = ""
        if summary.incomplete or summary.failed:
            note = f"{summary.incomplete} fiches incomplètes (champs en panne LLM : relancez la même commande), {summary.failed} en échec"
        return StageResult(status="done", artifact=ARTIFACT, message=note)