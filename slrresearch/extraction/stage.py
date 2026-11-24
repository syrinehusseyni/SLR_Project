"""Étape 5a : lire les PDF téléchargés (étape 4), nettoyer, découper en chunks avec provenance. Aucun LLM.

PLAN (papiers `downloaded` de `documents`) -> EXECUTE (lecture + chunks, un fichier par papier)
-> OBSERVE (pages, caractères, pages vides, en-têtes répétés) -> DECIDE : `extracted` ou `failed` (raison écrite).

Reprise : un papier déjà extrait (même empreinte sha256 du PDF, mêmes réglages) n'est jamais relu. Le fichier de chaque
papier est écrit dès qu'il est prêt : Ctrl+C ne perd rien. L'artefact `extraction` (index) n'est écrit qu'à la fin.
Pas de porte humaine : un papier illisible est signalé `failed`, il ne bloque pas les autres.
"""
import time
from pathlib import Path

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound
from slrresearch.core.schemas.documents import DocumentRecord, DocumentsOutcome
from slrresearch.core.schemas.extraction import (
    ExtractedPaperEntry, ExtractionLimits, ExtractionOutcome, ExtractionParams, ExtractionSummary, PaperText,
)
from slrresearch.download import validate
from slrresearch.download.agent import safe_filename, sha256_hex
from slrresearch.extraction.reader import (
    ExtractionError, extract_paper, load_paper_text, paper_file, save_paper_text,
)
from slrresearch.extraction.text import CLEANING_VERSION, validate_chunking

ARTIFACT = "extraction"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def entry_from_text(text: PaperText, run_dir: Path, path: Path) -> ExtractedPaperEntry:
    return ExtractedPaperEntry(
        paper_id=text.paper_id, status="extracted", file=path.relative_to(run_dir).as_posix(),
        source_sha256=text.source_sha256, total_pages=text.total_pages, pages_read=text.pages_read,
        chars=sum(text.page_chars), chunks=len(text.chunks), empty_pages=sum(1 for n in text.page_chars if n == 0),
        repeated_lines=len(text.repeated_lines), truncated=text.truncated)


class ExtractionStage:
    name = "extraction"

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 max_pages: int | None = None, max_papers: int | None = None,
                 min_chars_per_page: float = validate.DEFAULT_MIN_CHARS_PER_PAGE, force: bool = False,
                 progress=None, clock=time.monotonic):
        validate_chunking(chunk_size, chunk_overlap)
        self.params = ExtractionParams(chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_pages=max_pages,
                                       min_chars_per_page=min_chars_per_page, cleaning=CLEANING_VERSION)
        self.max_papers = max_papers  # refuse de démarrer si plus de N papiers doivent être (ré)extraits
        self.force = force            # ignorer les fichiers déjà extraits
        self.progress = progress      # fonction (index, total, paper_id, statut) ; None = silencieux
        self.clock = clock

    def run(self, ctx) -> StageResult:
        try:
            documents = ctx.store.read_artifact("documents", DocumentsOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « documents » introuvable : terminez d'abord l'étape download.")
        open_g3 = [r.paper_id for r in documents.records if r.status == "manual_required"]
        if open_g3:
            return StageResult(status="failed", message=(
                f"{len(open_g3)} papiers sont encore en « manual_required » (point G3 ouvert) : terminez le téléchargement "
                "(python cli.py gates --run-id ...) avant d'extraire."))
        records = [r for r in documents.records if r.status == "downloaded"]
        skipped = len(documents.records) - len(records)
        if not records:
            return StageResult(status="failed", message="aucun PDF téléchargé : rien à extraire.")
        names = [safe_filename(r.paper_id) for r in records]
        if len(set(names)) != len(names):
            return StageResult(status="failed", message="deux papiers donneraient le même nom de fichier : identifiants à revoir.")

        run_dir = ctx.store.run_dir
        cached: dict[str, PaperText] = {}
        if not self.force:
            for record in records:
                old = load_paper_text(paper_file(run_dir, record.paper_id))
                if old is not None and old.source_sha256 == record.sha256 and old.params == self.params:
                    cached[record.paper_id] = old
        todo = [r for r in records if r.paper_id not in cached]
        ctx.trace.log("extraction", "started", downloaded=len(records), reused=len(cached), to_extract=len(todo))
        if self.max_papers is not None and len(todo) > self.max_papers:
            ctx.trace.log("extraction", "refused_too_many", to_extract=len(todo), max_papers=self.max_papers)
            return StageResult(status="failed", message=(
                f"{len(todo)} papiers doivent être extraits, plus que la limite --max-papers {self.max_papers}. "
                "Aucune extraction n'a été faite. Augmentez --max-papers."))

        entries: dict[str, ExtractedPaperEntry] = {
            pid: entry_from_text(text, run_dir, paper_file(run_dir, pid)) for pid, text in cached.items()}
        total = len(records)
        try:
            for record in todo:
                entries[record.paper_id] = self._extract_one(ctx, record)
                entry = entries[record.paper_id]
                self._tell(records.index(record) + 1, total, record.paper_id,
                           f"{entry.pages_read} pages, {entry.chunks} chunks" if entry.status == "extracted" else f"échec : {entry.reason}")
        except KeyboardInterrupt:
            ctx.trace.log("extraction", "interrupted", done=len(entries), total=total)
            raise  # les fichiers déjà écrits sont conservés : la reprise les réutilise
        return self._finalize(ctx, records, entries, skipped)

    def _extract_one(self, ctx, record: DocumentRecord) -> ExtractedPaperEntry:
        path = ctx.store.run_dir / record.pdf_path
        try:
            data = path.read_bytes()
        except OSError:
            return self._fail(ctx, record, "PDF introuvable ou illisible sur le disque : relancez l'étape download")
        if sha256_hex(data) != record.sha256:
            return self._fail(ctx, record, "le PDF sur le disque a changé depuis le téléchargement (empreinte différente) : relancez l'étape download")
        try:
            text = extract_paper(record.paper_id, data, self.params)
        except ExtractionError as exc:
            return self._fail(ctx, record, exc.reason)
        target = paper_file(ctx.store.run_dir, record.paper_id)
        save_paper_text(target, text)
        ctx.trace.log("extraction", "paper_done", paper_id=record.paper_id, pages=text.pages_read, chunks=len(text.chunks))
        return entry_from_text(text, ctx.store.run_dir, target)

    @staticmethod
    def _fail(ctx, record: DocumentRecord, reason: str) -> ExtractedPaperEntry:
        ctx.trace.log("extraction", "paper_failed", paper_id=record.paper_id, reason=reason)
        return ExtractedPaperEntry(paper_id=record.paper_id, status="failed", reason=reason, source_sha256=record.sha256)

    def _tell(self, index: int, total: int, paper_id: str, status: str) -> None:
        if self.progress is not None:
            self.progress(index, total, paper_id, status)

    def _finalize(self, ctx, records: list[DocumentRecord], entries: dict[str, ExtractedPaperEntry], skipped: int) -> StageResult:
        ordered = [entries[r.paper_id] for r in records]
        extracted = sum(1 for e in ordered if e.status == "extracted")
        summary = ExtractionSummary(total=len(ordered), extracted=extracted, failed=len(ordered) - extracted,
                                    chunks=sum(e.chunks for e in ordered), skipped_unavailable=skipped)
        limits = ExtractionLimits(**self.params.model_dump(), max_papers=self.max_papers)
        ctx.store.write_artifact(ARTIFACT, ExtractionOutcome, ExtractionOutcome(limits=limits, papers=ordered, summary=summary).model_dump())
        ctx.trace.log("extraction", "written", **summary.model_dump())
        failed = [e for e in ordered if e.status == "failed"]
        if extracted == 0:
            return StageResult(status="failed", message="aucun papier n'a pu être extrait : " + " ; ".join(f"{e.paper_id} ({e.reason})" for e in failed[:5]))
        note = f"{len(failed)} papiers non extraits : " + " ; ".join(f"{e.paper_id} ({e.reason})" for e in failed[:5]) if failed else ""
        return StageResult(status="done", artifact=ARTIFACT, message=note)