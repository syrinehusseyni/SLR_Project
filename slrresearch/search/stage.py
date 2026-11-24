"""Étape 2 : exécuter le plan de recherche approuvé en G1, dédoublonner, écrire l'artefact `search`.

PLAN (fait en 1b, validé en G1) -> EXECUTE (ici, borné par un budget d'appels HTTP) -> OBSERVE (chaque requête
a un statut) -> DECIDE : une source en panne n'arrête pas le run ; zéro résultat au total => échec.
Aucun LLM : le programme exécute le plan que l'humain a validé. Pas de porte humaine propre à cette étape.
"""
import time

import httpx

from slrresearch.core.budgets import Budget
from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.core.schemas.search import QueryRun, SearchOutcome, SearchSummary
from slrresearch.search.dedup import deduplicate
from slrresearch.search.http import HttpClient, MissingCredential, RateLimiter, SearchError
from slrresearch.search.sources import SearchConfig, run_query

ARTIFACT = "search"
G1_ITEM_ID = "G1:protocol:main"
APPROVED = "approved"


class SearchStage:
    name = "search"

    def __init__(self, config: SearchConfig, results_per_query: int = 50, max_http_calls: int = 80,
                 force: bool = False, transport=None, sleep=time.sleep, clock=time.monotonic):
        self.config = config
        self.results_per_query = results_per_query
        self.max_http_calls = max_http_calls
        self.force = force
        self.transport = transport  # httpx.MockTransport dans les tests
        self.sleep = sleep
        self.clock = clock

    def run(self, ctx) -> StageResult:
        try:
            protocol = ctx.store.read_artifact("protocol", Protocol)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « protocol » introuvable : terminez d'abord l'étape protocol.")
        if not self._g1_approved(ctx):
            return StageResult(status="failed", message="la porte G1 n'est pas approuvée : fermez-la (cli.py close) puis relancez protocol.")
        if ctx.store.exists(ARTIFACT) and not self.force:
            ctx.trace.log("search", "reused")
            return StageResult(status="done", artifact=ARTIFACT)

        budget = Budget(max_calls=self.max_http_calls)
        limiter = RateLimiter(clock=self.clock, sleep=self.sleep)
        records: list[Paper] = []
        runs: list[QueryRun] = []
        with httpx.Client(transport=self.transport, timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
            http = HttpClient(client, budget, ctx.trace, limiter=limiter, sleep=self.sleep)
            stop = ""
            for sq in protocol.search_plan.queries:
                base = dict(source=sq.source, query_param=sq.query_param, query=sq.query)
                if stop:
                    runs.append(QueryRun(**base, status="skipped", error=stop))
                    continue
                try:
                    result = run_query(http, self.config, sq, self.results_per_query)
                except BudgetExceeded:
                    stop = f"budget de {self.max_http_calls} appels HTTP épuisé"
                    runs.append(QueryRun(**base, status="skipped", error=stop))
                    ctx.trace.log("search", "budget_exhausted", calls=budget.calls)
                    continue
                except MissingCredential as exc:
                    runs.append(QueryRun(**base, status="skipped", error=str(exc)))
                    ctx.trace.log("search", "skipped", source=sq.source, reason=str(exc))
                    continue
                except SearchError as exc:
                    status = "rate_limited" if exc.status == 429 else "error"
                    runs.append(QueryRun(**base, status=status, error=str(exc)))
                    ctx.trace.log("search", status, source=sq.source, error=str(exc))
                    continue
                records.extend(result.papers)
                runs.append(QueryRun(**base, status="ok", retrieved=len(result.papers), skipped_invalid=result.skipped))
                ctx.trace.log("search", "query_ok", source=sq.source, retrieved=len(result.papers), skipped=result.skipped)

        if not records:
            reasons = "; ".join(sorted({r.error for r in runs if r.error})) or "aucune notice exploitable"
            ctx.trace.log("search", "failed", error=reasons)
            return StageResult(status="failed", message=f"aucun résultat récupéré : {reasons}")

        dedup = deduplicate(records)
        per_source: dict[str, int] = {}
        for paper in records:
            per_source[paper.source] = per_source.get(paper.source, 0) + 1
        summary = SearchSummary(
            per_source=per_source,
            retrieved=len(records),
            duplicates_removed=dedup.duplicates_removed,
            unique=len(dedup.papers),
            without_abstract=sum(1 for p in dedup.papers if not p.abstract.strip()),
        )
        outcome = SearchOutcome(runs=runs, papers=dedup.papers, found_in=dedup.found_in, summary=summary)
        ctx.store.write_artifact(ARTIFACT, SearchOutcome, outcome.model_dump())
        ctx.trace.log("search", "written", **summary.model_dump())
        return StageResult(status="done", artifact=ARTIFACT)

    @staticmethod
    def _g1_approved(ctx) -> bool:
        for item in ctx.queue.list_items(status="closed", gate="G1"):
            if item.id == G1_ITEM_ID:
                return item.decision == APPROVED
        return False