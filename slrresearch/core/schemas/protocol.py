# Pas de "from __future__ import annotations" ici (voir papers.py).
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ResearchQuestion(BaseModel):
    """Contrat d'une question de recherche (meme contrat que le TD02)."""

    identifier: str = Field(pattern=r"^RQ[0-9]+$")
    question: str = Field(min_length=15)


class QuestionsOutcome(BaseModel):
    """Resultat de la generation des questions : c'est l'artefact ecrit par l'etape."""

    topic: str = Field(min_length=1)
    status: Literal["accepted", "repaired", "human_required"]
    text: str  # derniere sortie brute du LLM (gardee pour comprendre un echec)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    error: str = ""
    repair_attempts: int = 0

    @model_validator(mode="after")
    def _coherent(self):
        if self.status in ("accepted", "repaired") and not self.questions:
            raise ValueError("un resultat accepte/repare doit contenir des questions")
        if self.status == "human_required" and (self.questions or not self.error):
            raise ValueError("'human_required' : pas de questions et une erreur expliquee")
        return self