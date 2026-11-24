# ATTENTION : pas de "from __future__ import annotations" dans ce fichier
# (cela casse la résolution de Literal dans les modèles Pydantic).
"""Schémas du protocole complet (étape 1b).

Les schémas de l'étape 1a (ResearchQuestion, QuestionsOutcome) restent dans protocol.py ;
ce fichier ajoute : concepts de recherche, critères, facettes, requêtes par source, Protocol.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from slrresearch.core.schemas.protocol import ResearchQuestion

SourceName = Literal["openalex", "crossref", "semanticscholar", "arxiv"]
SectionStatus = Literal["accepted", "repaired"]

# Mots interdits comme termes de recherche : ils seraient lus comme des opérateurs.
_OPERATORS = {"AND", "OR", "NOT", "ANDNOT"}
_FORBIDDEN_CHARS = '"()'


class SearchConcept(BaseModel):
    """Un groupe de synonymes. Dans la requête : termes reliés par OR, concepts reliés par AND."""

    id: str = Field(pattern=r"^C[0-9]+$")
    label: str = Field(min_length=2)
    terms: list[str] = Field(min_length=2, max_length=8)

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("terms")
    @classmethod
    def _terms(cls, terms: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in terms:
            term = " ".join(raw.split())
            if len(term) < 2:
                raise ValueError(f"terme trop court : {raw!r}")
            if any(ch in term for ch in _FORBIDDEN_CHARS):
                raise ValueError(f"terme avec guillemets ou parenthèses : {raw!r}")
            if term.upper() in _OPERATORS:
                raise ValueError(f"terme interdit (opérateur) : {raw!r}")
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(term)
        if len(cleaned) < 2:
            raise ValueError("il faut au moins 2 termes distincts par concept")
        return cleaned


class Criterion(BaseModel):
    id: str = Field(pattern=r"^[IE][0-9]+$")
    kind: Literal["include", "exclude"]
    text: str = Field(min_length=10)

    @model_validator(mode="after")
    def _id_matches_kind(self):
        expected = "I" if self.kind == "include" else "E"
        if not self.id.startswith(expected):
            raise ValueError(f"l'identifiant {self.id} ne correspond pas au type {self.kind}")
        return self


class Criteria(BaseModel):
    inclusion: list[Criterion] = Field(min_length=1)
    exclusion: list[Criterion] = Field(min_length=1)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    languages: list[str] = Field(default_factory=lambda: ["en"])

    @model_validator(mode="after")
    def _consistent(self):
        if any(c.kind != "include" for c in self.inclusion):
            raise ValueError("inclusion ne peut contenir que des critères 'include'")
        if any(c.kind != "exclude" for c in self.exclusion):
            raise ValueError("exclusion ne peut contenir que des critères 'exclude'")
        ids = [c.id for c in self.inclusion + self.exclusion]
        if len(ids) != len(set(ids)):
            raise ValueError("identifiants de critères en double")
        return self


class FacetSpec(BaseModel):
    """Information requise pour répondre à une question (design, échantillon, métrique...).

    À l'étape 7, une facette ne sera « couverte » que par une citation vérifiée.
    """

    id: str = Field(pattern=r"^F[0-9]+$")
    rq_id: str = Field(pattern=r"^RQ[0-9]+$")
    label: str = Field(min_length=2)
    description: str = Field(min_length=5)
    blocking: bool = True


class SourceQuery(BaseModel):
    """Une requête prête à envoyer à une source : `query_param` = `query`, plus `params`.

    La valeur n'est PAS encodée pour l'URL : le client HTTP (étape 2) s'en charge.
    """

    source: SourceName
    query_param: str = Field(min_length=1)
    query: str = Field(min_length=3)
    params: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class SearchPlan(BaseModel):
    queries: list[SourceQuery] = Field(min_length=1)


class Protocol(BaseModel):
    """Artefact `protocol` : ce que l'humain valide à la porte G1."""

    topic: str
    questions: list[ResearchQuestion] = Field(min_length=1)
    concepts: list[SearchConcept] = Field(min_length=1)
    criteria: Criteria
    facets: list[FacetSpec] = Field(min_length=1)
    search_plan: SearchPlan
    generation: dict[str, SectionStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self):
        rq_ids = [q.identifier for q in self.questions]
        if len(rq_ids) != len(set(rq_ids)):
            raise ValueError("questions en double")
        concept_ids = [c.id for c in self.concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("concepts en double")
        facet_ids = [f.id for f in self.facets]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("facettes en double")
        unknown = {f.rq_id for f in self.facets} - set(rq_ids)
        if unknown:
            raise ValueError(f"facettes liées à des questions inconnues : {sorted(unknown)}")
        uncovered = set(rq_ids) - {f.rq_id for f in self.facets}
        if uncovered:
            raise ValueError(f"questions sans aucune facette : {sorted(uncovered)}")
        return self