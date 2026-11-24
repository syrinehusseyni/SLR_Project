# Pas de "from __future__ import annotations" ici (voir papers.py).
from typing import Literal

from pydantic import BaseModel, Field


class StageStatus(BaseModel):
    """Etat d'UNE etape du pipeline."""

    name: str = Field(min_length=1)
    state: Literal["pending", "running", "waiting_human", "done", "failed"] = "pending"
    artifact: str = ""  # nom de l'artefact principal produit
    started_at: str = ""
    finished_at: str = ""
    error: str = ""


class RunState(BaseModel):
    """Etat d'un run complet : ecrit dans runs/<id>/artifacts/state.json."""

    run_id: str = Field(min_length=1)
    topic: str = ""
    stages: dict[str, StageStatus] = Field(default_factory=dict)