"""Étape 4 : télécharger et valider le PDF des papiers INCLUS à l'étape 3, puis porte G3 pour les autres.

PLAN (papiers inclus du screening) -> EXECUTE (agent de téléchargement, borné par un budget d'appels HTTP)
-> OBSERVE (chaque tentative est tracée) -> DECIDE : PDF valide => `downloaded` ; sinon `manual_required` et
un point G3 par papier. Aucun LLM. Pas d'OCR : un PDF sans texte va en revue humaine.

Porte G3 (décisions humaines acceptées) :
  - `provided`    : l'humain a déposé le PDF dans runs/<id>/documents/manual/<paper_id>.pdf ; il doit être un vrai
                    PDF lisible avec une couche texte (le contrôle du titre est levé : l'humain garantit l'article) ;
  - `unavailable` : l'humain n'a pas trouvé le PDF ; le papier reste dans la revue mais sans texte intégral.
Politique `on_missing` (réglage d'exécution, enregistré dans l'artefact) : `review` (défaut de la bibliothèque) ouvre un point G3 par
papier sans PDF ; `unavailable` déclare ces papiers indisponibles **sans intervention humaine** (décision tracée : résolveur `policy`,
et les points G3 déjà ouverts sont fermés par « automatique »). Une décision humaine déjà prise en G3 prime toujours.
Reprise : l'état de chaque papier est gardé dans l'artefact `documents_progress` ; un PDF déjà validé (empreinte
sha256 intacte) n'est jamais retéléchargé. L'artefact `documents` n'est écrit que lorsque TOUS les papiers sont traités.
"""
import time
from pathlib import Path

import httpx

from slrresearch.core.budgets import Budget
from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactInvalid, ArtifactNotFound, BudgetExceeded
from slrresearch.core.schemas.documents import (
    DocumentRecord, DocumentsOutcome, DocumentsProgress, DocumentsSummary, DownloadAttempt, DownloadLimits,
)
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.screening import ScreeningOutcome
from slrresearch.core.schemas.search import SearchOutcome
from slrresearch.download import validate
from slrresearch.download.agent import DownloadSettings, download_paper, safe_filename, store_pdf, stored_file_ok
from slrresearch.download.resolvers import DOWNLOAD_INTERVALS
from slrresearch.search.http import HttpClient, RateLimiter
from slrresearch.search.sources import SearchConfig

ARTIFACT = "documents"
PROGRESS = "documents_progress"
G3_DECISIONS = ("provided", "unavailable")
MISSING_POLICIES = ("review", "unavailable")
POLICY_RESOLVER = "policy"
POLICY_VALIDATION = "unavailable:policy"
POLICY_AUTHOR = "automatique (on_missing=unavailable)"
POLICY_REASON = ("politique d'exécution on_missing=unavailable : aucun PDF en accès libre téléchargeable automatiquement ; "
                 "papier déclaré indisponible sans intervention humaine")

MANUAL_PROBLEMS = {
    "missing_file": "aucun fichier à l'emplacement attendu",
    "not_pdf": "le fichier n'est pas un PDF",
    "unreadable": "le PDF est illisible ou protégé par mot de passe",
    "no_text_layer": "le PDF n'a pas de couche texte (scanné) : sans OCR, il est inutilisable",
    "too_large": "le PDF dépasse la taille maximale",
}


def manual_pdf_path(store, paper_id: str) -> Path:
    return store.run_dir / "documents" / "manual" / (safe_filename(paper_id) + ".pdf")


def check_manual_pdf(store, paper: Paper, settings: DownloadSettings) -> tuple[str, bytes | None]:
    """Contrôle du PDF déposé par l'humain. ('', octets) si utilisable, sinon (verdict, None). Titre non contrôlé."""
    path = manual_pdf_path(store, paper.id)
    if not path.is_file():
        return "missing_file", None
    try:
        data = path.read_bytes()
    except OSError:
        return "unreadable", None
    check = validate.validate_pdf(
        data, paper.title, max_bytes=settings.max_bytes, min_chars_per_page=settings.min_chars_per_page,
        min_title_coverage=0.0, sample_pages=settings.sample_pages)
    return ("" if check.ok else check.verdict), (data if check.ok else None)


def manual_pdf_problem(store, paper: Paper, settings: DownloadSettings) -> str:
    """Phrase d'explication si le PDF déposé est inutilisable, '' s'il est bon (utilisé par `cli.py close`)."""
    verdict, _ = check_manual_pdf(store, paper, settings)
    if not verdict:
        return ""
    return f"{MANUAL_PROBLEMS.get(verdict, verdict)} ({manual_pdf_path(store, paper.id)})"


def looks_unreachable(record: DocumentRecord) -> bool:
    """Toutes les tentatives ont échoué faute de réseau (pas de code HTTP) : on ne conclut rien sur le papier."""
    def network(a: DownloadAttempt) -> bool:
        return (a.http_status is None and a.validation in ("http_error", "lookup:network_error")) or \
            a.validation.startswith("network_error")
    return record.status == "manual_required" and bool(record.attempts) and all(network(a) for a in record.attempts)


class DownloadStage:
    name = "download"

    def __init__(self, config: SearchConfig, settings: DownloadSettings | None = None, max_papers: int | None = None,
                 pause: float = 1.0, batch: int | None = None, max_http_calls: int = 400, force: bool = False,
                 retry_manual: bool = False, on_missing: str = "review", save_every: int = 5, max_unreachable: int = 3, progress=None,
                 transport=None, sleep=time.sleep, clock=time.monotonic):
        self.config = config
        self.settings = settings or DownloadSettings()
        self.max_papers = max_papers          # refuse de démarrer si plus de N papiers doivent être téléchargés
        self.pause = pause                    # secondes entre deux téléchargements (hors arXiv : 3 s imposées)
        self.batch = batch                    # nombre max de papiers téléchargés par lancement (None = pas de limite)
        self.max_http_calls = max_http_calls  # budget d'appels HTTP du lancement (recherches d'adresses + PDF)
        self.force = force                    # ignorer l'état précédent et tout retélécharger
        self.retry_manual = retry_manual      # retenter les `manual_required` sans décision G3
        if on_missing not in MISSING_POLICIES:
            raise ValueError(f"on_missing doit valoir l'une de {MISSING_POLICIES}, reçu : {on_missing!r}")
        self.on_missing = on_missing          # review = point G3 ; unavailable = papier déclaré indisponible automatiquement
        self.save_every = save_every
        self.max_unreachable = max_unreachable  # papiers de suite sans aucun accès réseau avant d'arrêter
        self.progress = progress              # fonction (index, total, paper_id, statut) ; None = silencieux
        self.transport = transport            # httpx.MockTransport dans les tests
        self.sleep = sleep
        self.clock = clock

    # -- point d'entrée ------------------------------------------------------
    def run(self, ctx) -> StageResult:
        try:
            screening = ctx.store.read_artifact("screening", ScreeningOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « screening » introuvable : terminez d'abord l'étape screening.")
        try:
            search = ctx.store.read_artifact("search", SearchOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « search » introuvable : terminez d'abord l'étape search.")
        if screening.summary.pending:
            return StageResult(status="failed", message=(
                f"{screening.summary.pending} points G2 sont encore ouverts : terminez le screening "
                "(python cli.py screening --run-id ...) avant de télécharger."))
        papers = {p.id: p for p in search.papers}
        included = [r.paper_id for r in screening.records if r.final == "include"]
        if not included:
            return StageResult(status="failed", message="aucun papier inclus au screening : rien à télécharger.")
        absent = [i for i in included if i not in papers]
        if absent:
            return StageResult(status="failed", message=f"papiers inclus absents de l'artefact search : {', '.join(absent[:5])}")

        records = {} if self.force else self._load_progress(ctx, included)
        problems: list[str] = []
        for pid in list(records):
            kept = self._reconcile(ctx, papers[pid], records[pid], problems)
            if kept is None:
                del records[pid]
            else:
                records[pid] = kept
        self._save_progress(ctx, records)

        to_download = [pid for pid in included if pid not in records]
        ctx.trace.log("download", "started", included=len(included), kept=len(records), to_download=len(to_download))
        if self.max_papers is not None and len(to_download) > self.max_papers:
            ctx.trace.log("download", "refused_too_many", to_download=len(to_download), max_papers=self.max_papers)
            return StageResult(status="failed", message=(
                f"{len(to_download)} papiers doivent être téléchargés, plus que la limite --max-papers {self.max_papers}. "
                "Aucun téléchargement n'a été fait. Augmentez --max-papers ou réduisez le corpus (étape screening)."))

        before = set(records)
        outcome = self._download_all(ctx, papers, included, to_download, records)
        if outcome is not None:
            return outcome
        for pid in [p for p in records if p not in before and records[p].status == "manual_required"]:
            records[pid] = self._reconcile(ctx, papers[pid], records[pid], problems, retry=False)  # décision G3 déjà fermée
        self._apply_policy(ctx, included, records)
        self._save_progress(ctx, records)
        return self._finalize(ctx, papers, included, records, problems)

    # -- téléchargement ------------------------------------------------------
    def _download_all(self, ctx, papers, included, to_download, records) -> StageResult | None:
        """Télécharge les papiers restants. Renvoie un StageResult d'arrêt, ou None si tout est traité."""
        if not to_download:
            return None
        total = len(included)
        budget = Budget(max_calls=self.max_http_calls)
        limiter = RateLimiter({**DOWNLOAD_INTERVALS, "pdf": self.pause}, clock=self.clock, sleep=self.sleep)
        docs_dir = ctx.store.run_dir / "documents"
        fresh = streak = 0
        try:
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
                http = HttpClient(client, budget, ctx.trace, limiter=limiter, sleep=self.sleep)
                for pid in to_download:
                    if self.batch is not None and fresh >= self.batch:
                        return self._stop(ctx, records, "batch_done", len(records), total, (
                            f"lot de {self.batch} papiers terminé : {len(records)} sur {total} traités. "
                            "Progression conservée : relancez la même commande pour continuer."))
                    record = download_paper(papers[pid], http, self.config, self.settings, docs_dir)
                    fresh += 1
                    index = included.index(pid) + 1
                    if looks_unreachable(record):  # rien n'est enregistré : le papier sera retenté
                        streak += 1
                        self._tell(index, total, pid, "réseau inaccessible")
                        if streak >= self.max_unreachable:
                            return self._stop(ctx, records, "network_down", len(records), total, (
                                f"réseau inaccessible : {streak} papiers de suite sans aucune réponse. "
                                "Vérifiez votre connexion puis relancez la même commande (progression conservée)."))
                        continue
                    streak = 0
                    records[pid] = record
                    ctx.trace.log("download", "paper_done", paper_id=pid, status=record.status, attempts=len(record.attempts))
                    self._tell(index, total, pid, self._label(record))
                    if fresh % self.save_every == 0:
                        self._save_progress(ctx, records)
        except BudgetExceeded as exc:
            self._save_progress(ctx, records)
            ctx.trace.log("download", "budget_exhausted", done=len(records), total=total)
            return StageResult(status="failed", message=(
                f"budget épuisé après {len(records)} papiers sur {total} ({exc}) : progression conservée, "
                "relancez avec --max-calls plus grand."))
        except KeyboardInterrupt:
            self._save_progress(ctx, records)  # Ctrl+C : rien n'est perdu
            ctx.trace.log("download", "interrupted", done=len(records), total=total)
            raise
        self._save_progress(ctx, records)
        pending = [pid for pid in included if pid not in records]
        if pending:
            return StageResult(status="failed", message=(
                f"{len(pending)} papiers n'ont pas pu être contactés (réseau ou service indisponible) : "
                "relancez la même commande pour les reprendre."))
        return None

    def _stop(self, ctx, records, event: str, done: int, total: int, message: str) -> StageResult:
        self._save_progress(ctx, records)
        ctx.trace.log("download", event, done=done, total=total)
        return StageResult(status="failed", message=message)

    def _tell(self, index: int, total: int, paper_id: str, status: str) -> None:
        if self.progress is not None:
            self.progress(index, total, paper_id, status)

    @staticmethod
    def _label(record: DocumentRecord) -> str:
        if record.status == "downloaded":
            return "downloaded (" + next((a.resolver for a in reversed(record.attempts) if a.ok), "?") + ")"
        return record.status

    # -- reprise et décisions humaines (aucun réseau) ---------------------------
    def _load_progress(self, ctx, included: list[str]) -> dict[str, DocumentRecord]:
        try:
            progress = ctx.store.read_artifact(PROGRESS, DocumentsProgress)
        except (ArtifactNotFound, ArtifactInvalid):
            return {}
        return {pid: rec for pid, rec in progress.records.items() if pid in set(included)}

    @staticmethod
    def _save_progress(ctx, records: dict[str, DocumentRecord]) -> None:
        ctx.store.write_artifact(PROGRESS, DocumentsProgress, DocumentsProgress(records=records).model_dump())

    def _reconcile(self, ctx, paper: Paper, record: DocumentRecord, problems: list[str],
                   retry: bool = True) -> DocumentRecord | None:
        """Renvoie l'enregistrement à garder, ou None s'il faut (re)télécharger le papier."""
        if record.status == "downloaded":
            if stored_file_ok(record, ctx.store.run_dir):
                return record
            ctx.trace.log("download", "file_missing", paper_id=paper.id)
            return None
        if record.status == "unavailable":
            decided_by_policy = bool(record.attempts) and record.attempts[-1].resolver == POLICY_RESOLVER
            return None if (decided_by_policy and self.retry_manual and retry) else record
        item = ctx.queue.get(f"G3:paper:{paper.id}")
        if item is not None and item.status == "closed":
            choice = item.decision.strip().lower()
            if choice == "unavailable":
                return record.model_copy(update={"status": "unavailable"})
            if choice == "provided":
                return self._ingest_manual(ctx, paper, record, problems)
            problems.append(f"{item.id} fermé avec « {item.decision} » (attendu : provided ou unavailable)")
            return record
        return None if (self.retry_manual and retry) else record

    def _apply_policy(self, ctx, included: list[str], records: dict[str, DocumentRecord]) -> None:
        """Politique `unavailable` : un papier sans PDF valide et sans décision humaine est déclaré indisponible."""
        if self.on_missing != "unavailable":
            return
        for pid in included:
            record = records[pid]
            if record.status != "manual_required":
                continue
            item = ctx.queue.get(f"G3:paper:{pid}")
            if item is not None and item.status == "closed":
                continue  # une décision humaine existe déjà : elle prime (déjà traitée par _reconcile)
            attempt = DownloadAttempt(resolver=POLICY_RESOLVER, url="", validation=POLICY_VALIDATION)
            records[pid] = record.model_copy(update={"status": "unavailable", "attempts": [*record.attempts, attempt]})
            if item is not None:  # point ouvert lors d'un essai précédent : on le ferme, la décision est tracée
                ctx.queue.close_item(item.id, "unavailable", POLICY_AUTHOR, POLICY_REASON)
            ctx.trace.log("download", "unavailable_by_policy", paper_id=pid)

    def _ingest_manual(self, ctx, paper: Paper, record: DocumentRecord, problems: list[str]) -> DocumentRecord:
        verdict, data = check_manual_pdf(ctx.store, paper, self.settings)
        path = manual_pdf_path(ctx.store, paper.id)
        relative = f"documents/manual/{path.name}"
        if data is not None:
            pdf_path, digest = store_pdf(paper.id, data, ctx.store.run_dir / "documents")
            attempt = DownloadAttempt(resolver="manual", url=relative, validation="ok", ok=True)
            ctx.trace.log("download", "manual_ingested", paper_id=paper.id)
            return DocumentRecord(paper_id=paper.id, status="downloaded", attempts=[*record.attempts, attempt],
                                  pdf_path=pdf_path, sha256=digest, has_text_layer=True)
        problems.append(f"G3:paper:{paper.id} : PDF déposé inutilisable, {MANUAL_PROBLEMS.get(verdict, verdict)} ({path})")
        last = record.attempts[-1] if record.attempts else None
        if last is not None and last.resolver == "manual" and last.validation == verdict:
            return record
        attempt = DownloadAttempt(resolver="manual", url=relative, validation=verdict)
        return record.model_copy(update={"attempts": [*record.attempts, attempt]})

    # -- porte G3 et artefact ------------------------------------------------
    def _finalize(self, ctx, papers, included, records, problems) -> StageResult:
        ordered = [records[pid] for pid in included]
        open_items = []
        for record in ordered:
            if record.status != "manual_required":
                continue
            reason = self._reason(papers[record.paper_id], record)
            item = ctx.queue.open_item("G3", "paper", record.paper_id, reason)
            if item.status == "open":  # la file est en ajout seul : on affiche la raison ACTUELLE
                open_items.append(item.model_copy(update={"reason": reason}))

        by_resolver: dict[str, int] = {}
        for record in ordered:
            if record.status == "downloaded":
                name = next((a.resolver for a in reversed(record.attempts) if a.ok), "inconnu")
                by_resolver[name] = by_resolver.get(name, 0) + 1
        count = lambda status: sum(1 for r in ordered if r.status == status)  # noqa: E731
        summary = DocumentsSummary(total=len(ordered), downloaded=count("downloaded"), manual_required=count("manual_required"),
                                   unavailable=count("unavailable"), by_resolver=by_resolver)
        limits = DownloadLimits(max_papers=self.max_papers, on_missing=self.on_missing, max_mb=self.settings.max_bytes / (1024 * 1024),
                                max_http_calls=self.max_http_calls, pause=self.pause,
                                min_title_coverage=self.settings.min_title_coverage,
                                min_chars_per_page=self.settings.min_chars_per_page)
        ctx.store.write_artifact(ARTIFACT, DocumentsOutcome,
                                 DocumentsOutcome(limits=limits, records=ordered, summary=summary).model_dump())
        ctx.trace.log("download", "written", **summary.model_dump())
        if problems:
            return StageResult(status="failed", message="décision humaine à corriger : " + " ; ".join(problems))
        if open_items:
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=open_items)
        return StageResult(status="done", artifact=ARTIFACT)

    @staticmethod
    def _reason(paper: Paper, record: DocumentRecord) -> str:
        tried = "; ".join(f"{a.resolver}={a.validation or a.http_status}" for a in record.attempts[-4:])
        return f"Aucun PDF valide ({tried or 'aucune adresse trouvée'}) — {paper.title[:80]}"