"""Étape 1b : protocole complet + porte G1.

Premier passage : génère concepts, critères, facettes, construit les requêtes, écrit `protocol`,
ouvre G1 et renvoie `waiting_human`.
Reprise : G1 encore ouverte -> `waiting_human` ; G1 approuvée -> les requêtes sont RECONSTRUITES depuis les
concepts (que l'humain a pu corriger dans protocol.json) puis `done` ; G1 refusée -> `failed`.
"""
from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded, LLMError, SLRError
from slrresearch.core.schemas.protocol import QuestionsOutcome
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.protocol.concepts import generate_concepts
from slrresearch.protocol.criteria import generate_criteria
from slrresearch.protocol.extraction_schema import generate_facets
from slrresearch.protocol.search_plan import build_search_plan

ARTIFACT = "protocol"
GATE = "G1"
TARGET_TYPE = "protocol"
TARGET_ID = "main"
APPROVED = "approved"


class ProtocolGenerationFailed(SLRError):
    """Une section reste invalide après son repair (human_required)."""


def _checked(name: str, result):
    """S'arrête dès qu'une section reste invalide après son repair : pas d'appels LLM inutiles."""
    if result.status == "human_required":
        raise ProtocolGenerationFailed(f"section « {name} » invalide après un repair : {result.error}")
    return result


def build_protocol(llm, outcome: QuestionsOutcome, year_from=None, languages=("en",), trace=None) -> Protocol:
    """Trois appels LLM, dans cet ordre : concepts, critères, facettes."""
    topic, questions = outcome.topic, outcome.questions
    concepts = _checked("concepts", generate_concepts(llm, topic, questions, trace))
    criteria = _checked("criteria", generate_criteria(llm, topic, questions, year_from, languages, trace))
    facets = _checked("facets", generate_facets(llm, topic, questions, trace))
    return Protocol(
        topic=topic,
        questions=questions,
        concepts=concepts.items,
        criteria=criteria.items[0],
        facets=facets.items,
        search_plan=build_search_plan(concepts.items, year_from),
        generation={"concepts": concepts.status, "criteria": criteria.status, "facets": facets.status},
    )


class ProtocolStage:
    name = "protocol"

    def __init__(self, year_from: int | None = None, languages=("en",)):
        self.year_from = year_from
        self.languages = tuple(languages)

    def run(self, ctx) -> StageResult:
        try:
            outcome = ctx.store.read_artifact("questions", QuestionsOutcome)
        except ArtifactNotFound:
            return StageResult(status="failed", message="artefact « questions » introuvable : lancez d'abord l'étape questions.")
        if outcome.status == "human_required":
            return StageResult(status="failed", message="les questions ne sont pas valides : relancez l'étape questions.")

        if ctx.store.exists(ARTIFACT):
            return self._resume(ctx)

        try:
            protocol = build_protocol(ctx.llm("planner"), outcome, self.year_from, self.languages, ctx.trace)
        except ProtocolGenerationFailed as exc:
            ctx.trace.log("protocol", "failed", error=str(exc))
            return StageResult(status="failed", message=str(exc))
        except (LLMError, BudgetExceeded) as exc:
            message = f"appel LLM impossible : {exc}"
            ctx.trace.log("protocol", "failed", error=message)
            return StageResult(status="failed", message=message)

        ctx.store.write_artifact(ARTIFACT, Protocol, protocol.model_dump())
        ctx.trace.log("protocol", "written", concepts=len(protocol.concepts), facets=len(protocol.facets),
                      queries=len(protocol.search_plan.queries))
        item = self._open_gate(ctx)
        return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=[item])

    def _open_gate(self, ctx):
        # open_item est idempotent : un point déjà tranché est renvoyé tel quel, jamais redemandé.
        return ctx.queue.open_item(
            GATE, TARGET_TYPE, TARGET_ID,
            "Valider questions, critères d'inclusion/exclusion, concepts et facettes avant toute recherche "
            "(vous pouvez corriger artifacts/protocol.json avant d'approuver).",
        )

    def _resume(self, ctx) -> StageResult:
        item = self._open_gate(ctx)
        if item.status == "open":
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=[item])
        if item.decision != APPROVED:
            ctx.trace.log("protocol", "rejected", decision=item.decision, by=item.decided_by)
            return StageResult(
                status="failed",
                message=f"G1 refusée ({item.decision}) : {item.justification}. Créez un nouveau run ou corrigez protocol.json puis rouvrez la décision.",
            )
        # Approuvé : relit (revalide) le fichier éventuellement édité à la main, puis refait les requêtes.
        protocol = ctx.store.read_artifact(ARTIFACT, Protocol)
        year_from = protocol.criteria.year_from
        rebuilt = protocol.model_copy(update={"search_plan": build_search_plan(protocol.concepts, year_from)})
        ctx.store.write_artifact(ARTIFACT, Protocol, rebuilt.model_dump())
        ctx.trace.log("protocol", "approved", by=item.decided_by, queries=len(rebuilt.search_plan.queries))
        return StageResult(status="done", artifact=ARTIFACT)