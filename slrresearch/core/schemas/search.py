# ATTENTION : pas de "from __future__ import annotations" dans ce fichier.
"""Artefact `search` (étape 2) : ce qui a été trouvé, avec les chiffres de l'identification PRISMA."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import SourceName

RunStatus = Literal["ok", "error", "rate_limited", "skipped"]


class QueryRun(BaseModel):
    """Le sort d'UNE requête du plan de recherche."""

    source: SourceName
    query_param: str
    query: str
    status: RunStatus
    retrieved: int = Field(default=0, ge=0)
    skipped_invalid: int = Field(default=0, ge=0)  # notices reçues mais inutilisables (sans titre...)
    error: str = ""

    @model_validator(mode="after")
    def _coherent(self):
        if self.status != "ok" and not self.error:
            raise ValueError("une requête non réussie doit expliquer pourquoi (error)")
        if self.status != "ok" and self.retrieved:
            raise ValueError("une requête non réussie ne peut pas avoir de résultats")
        return self


class SearchSummary(BaseModel):
    per_source: dict[str, int]  # notices récupérées par source, avant dédoublonnage
    retrieved: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    unique: int = Field(ge=0)
    without_abstract: int = Field(ge=0)

    @model_validator(mode="after")
    def _arithmetic(self):
        if self.retrieved != self.unique + self.duplicates_removed:
            raise ValueError("retrieved doit valoir unique + duplicates_removed")
        if sum(self.per_source.values()) != self.retrieved:
            raise ValueError("la somme par source doit valoir retrieved")
        return self


class SearchOutcome(BaseModel):
    runs: list[QueryRun] = Field(min_length=1)
    papers: list[Paper] = Field(min_length=1)
    found_in: dict[str, list[str]]  # paper_id -> sources qui l'ont trouvé
    summary: SearchSummary

    @model_validator(mode="after")
    def _consistent(self):
        ids = [p.id for p in self.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("identifiants de papiers en double")
        if set(self.found_in) != set(ids):
            raise ValueError("found_in doit couvrir exactement les papiers")
        if self.summary.unique != len(self.papers):
            raise ValueError("summary.unique ne correspond pas au nombre de papiers")
        return self