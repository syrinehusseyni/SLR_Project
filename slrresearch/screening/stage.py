"""Étape 3 : screening titre/résumé. A (règles) + B (LLM) -> C (accord ou humain) -> porte G2.

PLAN (protocole approuvé en G1) -> EXECUTE (A sur tout, B sur les papiers qui le méritent) -> OBSERVE (statut LLM par
papier) -> DECIDE (accord => décision ; sinon porte G2). Un `uncertain` n'est jamais exporté comme décision.
Reprise : les verdicts LLM déjà obtenus sont gardés dans l'artefact `screening_llm` ; les décisions humaines
(points G2 fermés) sont réappliquées à chaque relance.
"""
import time

from slrresearch.core.errors import ArtifactInvalid, ArtifactNotFound, BudgetExceeded
from slrresearch.core.contract import StageResult
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.core.schemas.screening import (
    LLMResult, LLMResults, ScreeningOutcome, ScreeningRecord, ScreeningSummary,
)
from slrresearch.core.schemas.search import SearchOutcome
from slrresearch.screening.agreement import cohen_kappa
from slrresearch.screening.arbitration import DEFAULT_MIN_CONFIDENCE, decide
from slrresearch.screening.llm_screener import PROMPT_NAME, fingerprint, load_prompt, screen_paper
from slrresearch.screening.rules import apply_rules

ARTIFACT = "screening"
CACHE = "screening_llm"
G1_ITEM_ID = "G1:protocol:main"
APPROVED = "approved"
HUMAN_DECISIONS = ("include", "exclude")
CACHEABLE = ("ok", "grounding_failed", "invalid_format")  # une erreur réseau sera retentée à la relance


class ScreeningStage:
    name = "screening"

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE, force: bool = False,
                 prompt_template: str | None = None, save_every: int = 10,
                 batch: int | None = None, pause: float = 0.0, max_consecutive_errors: int = 3,
                 max_papers: int | None = None, retries: int = 0, retry_wait: float = 20.0,
                 progress=None, sleep=time.sleep):
        self.min_confidence = min_confidence
        self.force = force
        self.prompt_template = prompt_template  # tests : évite de lire le fichier
        self.save_every = save_every
        self.batch = batch  # nombre max de NOUVEAUX appels LLM par lancement (None = pas de limite)
        self.pause = pause  # secondes d'attente entre deux appels LLM (limite de débit du fournisseur)
        self.max_consecutive_errors = max_consecutive_errors
        self.max_papers = max_papers  # refuse de démarrer si plus de N papiers doivent passer par le LLM
        self.retries = retries  # nouveaux essais d'un papier dont l'appel LLM a échoué (limite de débit...)
        self.retry_wait = retry_wait  # attente avant le 1er nouvel essai ; multipliée à chaque essai suivant
        self.progress = progress  # fonction (index, total, paper_id, statut) ; None = silencieux
        self.sleep = sleep

    # -- point d'entrée ------------------------------------------------------
    def run(self, ctx) -> StageResult:
        try:
            protocol = ctx.store.read_artifact("protocol", Protocol)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « protocol » introuvable : terminez d'abord l'étape protocol.")
        try:
            search = ctx.store.read_artifact("search", SearchOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « search » introuvable : terminez d'abord l'étape search.")
        if not self._g1_approved(ctx):
            return StageResult(status="failed", message="la porte G1 n'est pas approuvée : le screening applique des critères non validés.")

        papers = {p.id: p for p in search.papers}
        if ctx.store.exists(ARTIFACT) and not self.force:
            outcome = ctx.store.read_artifact(ARTIFACT, ScreeningOutcome)
            ctx.trace.log("screening", "resumed")
            return self._finalize(ctx, outcome.topic, outcome.prompt, outcome.fingerprint, outcome.min_confidence,
                                  outcome.records, papers)

        template = self.prompt_template if self.prompt_template is not None else load_prompt()
        fp = fingerprint(template, protocol)
        cache = self._load_cache(ctx, fp)
        ctx.trace.log("screening", "started", papers=len(papers), prompt=PROMPT_NAME, fingerprint=fp, cached=len(cache.results))

        to_screen = sum(1 for p in search.papers if self._needs_call(p, protocol, cache))
        if self.max_papers is not None and to_screen > self.max_papers:
            ctx.trace.log("screening", "refused_too_many", to_screen=to_screen, max_papers=self.max_papers)
            return StageResult(status="failed", message=(
                f"{to_screen} papiers doivent passer par le LLM, plus que la limite --max-papers {self.max_papers}. "
                "Aucun appel LLM n'a été fait. Réduisez la recherche (python cli.py search --run-id ... --limit 10 --force) "
                "ou augmentez --max-papers."))

        records: list[ScreeningRecord] = []
        total = len(papers)
        fresh_calls = 0
        consecutive_errors = 0
        try:
            for index, paper in enumerate(search.papers, 1):
                needs_call = self._needs_call(paper, protocol, cache)
                if needs_call and self.batch is not None and fresh_calls >= self.batch:
                    return self._stop(ctx, cache, "batch_done", done=len(records), total=total, message=(
                        f"lot de {self.batch} appels LLM terminé : {len(records)} papiers sur {total} parcourus. "
                        "Progression conservée : relancez la même commande pour continuer."))
                if needs_call and fresh_calls > 0 and self.pause > 0:
                    self.sleep(self.pause)
                record = self._screen_one(ctx, paper, protocol, template, cache)
                attempt = 0
                while needs_call and record.llm_status == "llm_error" and attempt < self.retries:
                    attempt += 1
                    wait = self.retry_wait * attempt
                    if self.progress is not None:
                        self.progress(index, total, paper.id, f"erreur LLM, nouvel essai {attempt}/{self.retries} dans {wait:.0f}s")
                    ctx.trace.log("screening", "retry", paper_id=paper.id, attempt=attempt, wait=wait)
                    self.sleep(wait)
                    record = self._screen_one(ctx, paper, protocol, template, cache)
                records.append(record)
                if needs_call:
                    fresh_calls += 1
                    consecutive_errors = consecutive_errors + 1 if record.llm_status == "llm_error" else 0
                    if fresh_calls % self.save_every == 0:
                        self._save_cache(ctx, cache)
                if self.progress is not None:
                    self.progress(index, total, paper.id, record.llm_status)
                if consecutive_errors >= self.max_consecutive_errors:
                    return self._stop(ctx, cache, "llm_unavailable", done=len(records), total=total, message=(
                        f"LLM indisponible : {consecutive_errors} erreurs de suite (limite de débit ou service arrêté). "
                        f"{len(records)} papiers sur {total} parcourus, progression conservée. "
                        "Relancez plus tard, ou avec --pause pour ralentir le rythme."))
        except BudgetExceeded as exc:
            self._save_cache(ctx, cache)
            ctx.trace.log("screening", "budget_exhausted", done=len(records), total=total)
            return StageResult(status="failed", message=(
                f"budget épuisé après {len(records)} papiers sur {total} ({exc}) : "
                "les verdicts LLM déjà obtenus sont conservés, relancez avec un budget plus grand."))
        except KeyboardInterrupt:
            self._save_cache(ctx, cache)  # Ctrl+C : rien n'est perdu
            ctx.trace.log("screening", "interrupted", done=len(records), total=total)
            raise
        self._save_cache(ctx, cache)
        return self._finalize(ctx, protocol.topic, PROMPT_NAME, fp, self.min_confidence, records, papers)

    def _stop(self, ctx, cache: LLMResults, event: str, done: int, total: int, message: str) -> StageResult:
        self._save_cache(ctx, cache)
        ctx.trace.log("screening", event, done=done, total=total)
        return StageResult(status="failed", message=message)

    @staticmethod
    def _needs_call(paper: Paper, protocol: Protocol, cache: LLMResults) -> bool:
        """Vrai si ce papier va coûter un NOUVEL appel LLM (ni règle décisive, ni sans résumé, ni déjà en cache)."""
        rule = apply_rules(paper, protocol)
        return not (rule.decisive or not paper.abstract.strip()) and paper.id not in cache.results

    # -- un papier -----------------------------------------------------------
    def _screen_one(self, ctx, paper: Paper, protocol: Protocol, template: str, cache: LLMResults) -> ScreeningRecord:
        rule = apply_rules(paper, protocol)
        if rule.decisive or not paper.abstract.strip():
            result = LLMResult(paper_id=paper.id, status="not_run", note=rule.reasons[0])
        elif paper.id in cache.results:
            result = cache.results[paper.id]
        else:
            result = screen_paper(ctx.llm("screening"), paper, protocol, template, ctx.trace)
            if result.status in CACHEABLE:
                cache.results[paper.id] = result
        final, origin, escalation = decide(rule, result.status, result.verdict, self.min_confidence)
        return ScreeningRecord(paper_id=paper.id, title=paper.title, rule=rule, llm_status=result.status,
                               llm=result.verdict, final=final, origin=origin, escalation=escalation)

    # -- cache LLM -----------------------------------------------------------
    def _load_cache(self, ctx, fp: str) -> LLMResults:
        try:
            cache = ctx.store.read_artifact(CACHE, LLMResults)
        except (ArtifactNotFound, ArtifactInvalid):
            return LLMResults(fingerprint=fp)
        if cache.fingerprint != fp:
            ctx.trace.log("screening", "cache_discarded", reason="prompt, sujet ou critères modifiés")
            return LLMResults(fingerprint=fp)
        return cache

    @staticmethod
    def _save_cache(ctx, cache: LLMResults) -> None:
        ctx.store.write_artifact(CACHE, LLMResults, cache.model_dump())

    # -- porte G2 et artefact ------------------------------------------------
    def _finalize(self, ctx, topic: str, prompt: str, fp: str, min_conf: float,
                  records: list[ScreeningRecord], papers: dict[str, Paper]) -> StageResult:
        updated: list[ScreeningRecord] = []
        open_items = []
        problems: list[str] = []
        for record in records:
            if record.origin != "pending":
                updated.append(record)
                continue
            item = ctx.queue.open_item("G2", "paper", record.paper_id, f"{record.escalation} — {record.title[:80]}")
            if item.status == "open":
                # La file est en ajout seul : la raison d'un point déjà ouvert date de sa création. On affiche la raison
                # ACTUELLE (celle du verdict de ce run) pour ne pas induire l'humain en erreur.
                open_items.append(item.model_copy(update={"reason": f"{record.escalation} — {record.title[:80]}"}))
                updated.append(record)
                continue
            choice = item.decision.strip().lower()
            if choice not in HUMAN_DECISIONS:
                problems.append(f"{item.id} fermé avec « {item.decision} » (attendu : include ou exclude)")
                updated.append(record)
                continue
            updated.append(record.model_copy(update={
                "final": choice, "origin": "human", "human_by": item.decided_by, "human_why": item.justification}))

        summary = self._summary(updated, papers)
        outcome = ScreeningOutcome(topic=topic, prompt=prompt, fingerprint=fp, min_confidence=min_conf,
                                   records=updated, summary=summary)
        ctx.store.write_artifact(ARTIFACT, ScreeningOutcome, outcome.model_dump())
        ctx.trace.log("screening", "written", **summary.model_dump(exclude={"agreement"}),
                      kappa=summary.agreement.kappa)
        if problems:
            return StageResult(status="failed", message="décision humaine invalide : " + " ; ".join(problems))
        if open_items:
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=open_items)
        return StageResult(status="done", artifact=ARTIFACT)

    @staticmethod
    def _summary(records: list[ScreeningRecord], papers: dict[str, Paper]) -> ScreeningSummary:
        by_origin: dict[str, int] = {}
        for r in records:
            by_origin[r.origin] = by_origin.get(r.origin, 0) + 1
        pairs = [(r.rule.decision, r.llm.decision) for r in records if r.llm_status == "ok" and r.llm is not None]
        return ScreeningSummary(
            total=len(records),
            included=sum(1 for r in records if r.final == "include"),
            excluded=sum(1 for r in records if r.final == "exclude"),
            pending=sum(1 for r in records if r.origin == "pending"),
            by_origin=by_origin,
            without_abstract=sum(1 for r in records if not papers[r.paper_id].abstract.strip()),
            llm_failures=sum(1 for r in records if r.llm_status in ("invalid_format", "llm_error", "grounding_failed")),
            agreement=cohen_kappa(pairs),
        )

    @staticmethod
    def _g1_approved(ctx) -> bool:
        for item in ctx.queue.list_items(status="closed", gate="G1"):
            if item.id == G1_ITEM_ID:
                return item.decision == APPROVED
        return False