# ATTENTION : pas de "from __future__ import annotations" dans ce fichier
# (cela casse la résolution de Literal dans les modèles Pydantic).
"""Schémas de l'étape 3 : verdicts des agents A (règles) et B (LLM), enregistrement final, artefact `screening`."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Decision = Literal["include", "exclude", "uncertain"]
LLMStatus = Literal["ok", "grounding_failed", "invalid_format", "llm_error", "not_run"]
Origin = Literal["rule", "agreement", "human", "pending"]


class RuleVerdict(BaseModel):
    """Verdict de l'agent A (déterministe). `decisive` : règle objective qui se passe du LLM (ex. année)."""

    paper_id: str = Field(min_length=1)
    decision: Decision
    reasons: list[str] = Field(min_length=1)
    concepts_matched: list[str] = Field(default_factory=list)
    decisive: bool = False


class ScreeningVerdict(BaseModel):
    """Verdict de l'agent B (LLM). `paper_id` est imposé par l'application, jamais par le LLM."""

    paper_id: str = Field(min_length=1)
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    criteria: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(min_length=1)
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value):
        """Accepte 0..1 ou un pourcentage 0..100."""
        number = float(value)
        if 1 < number <= 100:
            return number / 100
        return number


class LLMResult(BaseModel):
    """Ce que B a donné pour un papier (mis en cache pour la reprise)."""

    paper_id: str = Field(min_length=1)
    status: LLMStatus
    verdict: ScreeningVerdict | None = None
    note: str = ""

    @model_validator(mode="after")
    def _coherent(self):
        needs_verdict = self.status in ("ok", "grounding_failed")
        if needs_verdict and self.verdict is None:
            raise ValueError(f"le statut {self.status} exige un verdict")
        if not needs_verdict and self.verdict is not None:
            raise ValueError(f"le statut {self.status} ne peut pas avoir de verdict")
        return self


class LLMResults(BaseModel):
    """Cache `screening_llm` : invalidé si le prompt, le sujet ou les critères changent (`fingerprint`)."""

    fingerprint: str
    results: dict[str, LLMResult] = Field(default_factory=dict)


class ScreeningRecord(BaseModel):
    paper_id: str = Field(min_length=1)
    title: str = ""
    rule: RuleVerdict
    llm_status: LLMStatus
    llm: ScreeningVerdict | None = None
    final: Decision
    origin: Origin
    escalation: str = ""
    human_by: str = ""
    human_why: str = ""

    @model_validator(mode="after")
    def _coherent(self):
        if self.origin == "pending" and self.final != "uncertain":
            raise ValueError("un point en attente n'a pas de décision finale")
        if self.origin != "pending" and self.final == "uncertain":
            raise ValueError("un 'uncertain' ne peut jamais être une décision finale exportée")
        if self.origin == "agreement" and self.llm_status != "ok":
            raise ValueError("un accord exige un verdict LLM valide")
        if self.origin == "human" and not (self.human_by and self.human_why):
            raise ValueError("une décision humaine exige son auteur et sa justification")
        if self.origin == "pending" and not self.escalation:
            raise ValueError("un point en attente doit dire pourquoi")
        return self


class AgreementStats(BaseModel):
    n: int = Field(ge=0)  # papiers où A et B ont tous deux rendu un verdict
    agreed: int = Field(ge=0)
    observed: float | None = None
    kappa: float | None = None  # None si indéfini (voir note)
    matrix: dict[str, int] = Field(default_factory=dict)  # "A|B" -> effectif
    note: str = ""


class ScreeningSummary(BaseModel):
    total: int = Field(ge=0)
    included: int = Field(ge=0)
    excluded: int = Field(ge=0)
    pending: int = Field(ge=0)
    by_origin: dict[str, int]
    without_abstract: int = Field(ge=0)
    llm_failures: int = Field(ge=0)
    agreement: AgreementStats

    @model_validator(mode="after")
    def _arithmetic(self):
        if self.included + self.excluded + self.pending != self.total:
            raise ValueError("included + excluded + pending doit valoir total")
        if sum(self.by_origin.values()) != self.total:
            raise ValueError("by_origin doit sommer à total")
        return self


class ScreeningOutcome(BaseModel):
    """Artefact `screening`."""

    topic: str
    prompt: str
    fingerprint: str
    min_confidence: float = Field(ge=0, le=1)
    records: list[ScreeningRecord] = Field(min_length=1)
    summary: ScreeningSummary

    @model_validator(mode="after")
    def _consistent(self):
        ids = [r.paper_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("papiers en double")
        s = self.summary
        if s.total != len(self.records):
            raise ValueError("summary.total ne correspond pas aux enregistrements")
        if s.pending != sum(1 for r in self.records if r.origin == "pending"):
            raise ValueError("summary.pending incohérent")
        return self