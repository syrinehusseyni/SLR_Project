from typing import Literal

from pydantic import BaseModel, Field, model_validator


Entailment = Literal["yes", "partial", "no", "error", "not_checked"]
FidelityStatus = Literal["verified", "human_review", "rejected", "llm_error", "skipped"]


class FidelityReport(BaseModel):
    paper_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)
    value: str = ""
    chunk_id: str = ""
    page: int | None = Field(default=None, ge=1)
    quote: str = ""
    quote_in_chunk: bool
    provenance_ok: bool
    numbers_ok: bool
    entailment: Entailment
    own_study: bool | None = None
    reason: str = ""
    judge_model: str = Field(min_length=1)
    input_fingerprint: str = ""
    human_decision: Literal["approved", "rejected"] | None = None
    reviewer: str = ""
    status: FidelityStatus


class FidelityEntry(BaseModel):
    paper_id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    reports: list[FidelityReport]


class FidelitySummary(BaseModel):
    papers: int = Field(ge=0)
    verified: int = Field(ge=0)
    human_review: int = Field(ge=0)
    rejected: int = Field(ge=0)
    llm_error: int = Field(ge=0)
    skipped: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)


class FidelityOutcome(BaseModel):
    version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    papers: list[FidelityEntry] = Field(min_length=1)
    summary: FidelitySummary

    @model_validator(mode="after")
    def _consistent(self):
        if self.summary.papers != len(self.papers):
            raise ValueError("summary.papers ne correspond pas aux fiches")
        reports = [r for p in self.papers for r in p.reports]
        for status in ("verified", "human_review", "rejected", "llm_error", "skipped"):
            if getattr(self.summary, status) != sum(r.status == status for r in reports):
                raise ValueError(f"summary.{status} incohérent")
        return self
