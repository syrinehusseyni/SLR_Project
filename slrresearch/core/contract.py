"""Contrat commun a TOUTES les etapes : meme entree (RunContext), meme sortie (StageResult).

Sans cette forme unique, l'orchestrateur devrait connaitre les details de chaque agent.
Avec elle, il lance n'importe quelle etape pareil, s'arrete a une porte ouverte et reprend.
"""
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from slrresearch.core.budgets import Budget
from slrresearch.core.config import Settings, build_model
from slrresearch.core.errors import ConfigError
from slrresearch.core.llm import LLMClient
from slrresearch.core.store import RunStore
from slrresearch.core.trace import Trace
from slrresearch.hitl.decisions import DecisionLog
from slrresearch.hitl.queue import ReviewItem, ReviewQueue


@dataclass
class RunContext:
    """Tout ce dont une etape a besoin. Une etape ne va rien chercher ailleurs."""

    store: RunStore
    trace: Trace
    queue: ReviewQueue
    settings: Settings
    budget: Budget | None = None
    llm_factory: Callable[[str], LLMClient] | None = None

    def llm(self, role: str) -> LLMClient:
        if self.llm_factory is None:
            raise ConfigError("RunContext sans llm_factory : impossible d'appeler un LLM")
        return self.llm_factory(role)


class StageResult(BaseModel):
    """Ce qu'une etape renvoie. Les evenements de trace sont ecrits directement dans la trace."""

    status: Literal["done", "waiting_human", "failed"]
    artifact: str = ""  # nom de l'artefact principal (dans runs/<id>/artifacts/)
    review_items: list[ReviewItem] = Field(default_factory=list)
    message: str = ""

    @model_validator(mode="after")
    def _coherent(self):
        if self.status == "done" and not self.artifact:
            raise ValueError("une etape 'done' doit produire un artefact")
        if self.status == "waiting_human" and not self.review_items:
            raise ValueError("une etape 'waiting_human' doit fournir au moins un point de revue")
        if self.status == "failed" and not self.message:
            raise ValueError("une etape 'failed' doit expliquer l'erreur dans 'message'")
        return self


@runtime_checkable
class Stage(Protocol):
    """Toute etape (protocole, recherche, screening...) respecte cette forme."""

    name: str

    def run(self, ctx: RunContext) -> StageResult: ...


def build_context(
    store: RunStore,
    settings: Settings,
    budget: Budget | None = None,
    make_model: Callable[[Settings, str], object] = build_model,
) -> RunContext:
    """Assemble le contexte d'un run. `make_model` est injectable : les tests y mettent un FakeLLM."""
    trace = Trace(store.trace_path)
    queue = ReviewQueue(store.hitl_path, DecisionLog(store.run_dir / "decisions.jsonl"))
    clients: dict[str, LLMClient] = {}  # un client par role : compteurs de tokens cumules

    def factory(role: str) -> LLMClient:
        if role not in clients:
            clients[role] = LLMClient(
                make_model(settings, role),
                model_name=settings.model_for(role),
                role=role,
                budget=budget,
                trace=trace,
            )
        return clients[role]

    return RunContext(store=store, trace=trace, queue=queue, settings=settings, budget=budget, llm_factory=factory)