"""Étape 1a sous forme de Stage : écrit l'artefact `questions` (c'est lui qui manquait dans runs/)."""
from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound, BudgetExceeded, LLMError
from slrresearch.core.schemas.protocol import QuestionsOutcome
from slrresearch.protocol.questions import generate_research_questions

ARTIFACT = "questions"


class QuestionsStage:
    name = "questions"

    def __init__(self, topic: str, n_questions: int = 3):
        self.topic = topic
        self.n_questions = n_questions

    def run(self, ctx) -> StageResult:
        # Reprise : un artefact déjà accepté/réparé est réutilisé (on ne repaie pas un appel LLM).
        try:
            existing = ctx.store.read_artifact(ARTIFACT, QuestionsOutcome)
        except ArtifactNotFound:
            existing = None
        if existing is not None and existing.status in ("accepted", "repaired"):
            ctx.trace.log("questions", "reused", status=existing.status)
            return StageResult(status="done", artifact=ARTIFACT)

        try:
            outcome = generate_research_questions(
                ctx.llm("planner"), self.topic, self.n_questions, trace=ctx.trace
            )
        except (LLMError, BudgetExceeded) as exc:
            message = f"appel LLM impossible : {exc}"
            ctx.trace.log("questions", "failed", error=message)
            return StageResult(status="failed", message=message)

        ctx.store.write_artifact(ARTIFACT, QuestionsOutcome, outcome.model_dump())
        ctx.trace.log("questions", "written", status=outcome.status, count=len(outcome.questions))
        if outcome.status == "human_required":
            return StageResult(
                status="failed",
                message=f"questions invalides après un repair : {outcome.error}. Relancez ou reformulez le sujet.",
            )
        return StageResult(status="done", artifact=ARTIFACT)