from typing import Literal

from pydantic import BaseModel, Field, model_validator

from slrresearch.core.schemas.fidelity import FidelityReport
from slrresearch.core.schemas.facet_plan import FacetCategory


EvidenceStatus = Literal["found", "not_reported", "missing", "human_review", "rejected", "llm_error"]


class FacetEvidenceRecord(BaseModel):
    paper_id: str = Field(min_length=1)
    facet_id: FacetCategory
    source_facet_ids: list[str] = Field(min_length=1)
    rq_ids: list[str] = Field(min_length=1)
    status: EvidenceStatus
    source_field: str = ""
    value: str = ""
    quote: str = ""
    chunk_id: str = ""
    page: int | None = Field(default=None, ge=1)
    reused: bool = False
    repair_attempted: bool = False
    note: str = ""
    input_fingerprint: str = ""
    fidelity: FidelityReport | None = None


class FacetEvidencePaper(BaseModel):
    paper_id: str = Field(min_length=1)
    evidence: list[FacetEvidenceRecord] = Field(min_length=1)


class FacetEvidenceSummary(BaseModel):
    papers: int = Field(ge=0)
    found: int = Field(ge=0)
    not_reported: int = Field(ge=0)
    missing: int = Field(ge=0)
    human_review: int = Field(ge=0)
    rejected: int = Field(ge=0)
    llm_error: int = Field(ge=0)
    reused: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)


class FacetEvidenceOutcome(BaseModel):
    version: str = Field(min_length=1)
    plan_hash: str = Field(min_length=64, max_length=64)
    judge_model: str = Field(min_length=1)
    extraction_model: str = Field(min_length=1)
    papers: list[FacetEvidencePaper] = Field(min_length=1)
    summary: FacetEvidenceSummary
    retrieval_mode: Literal["hybrid", "lexical", "semantic"] = "lexical"
    embedding_model: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.summary.papers != len(self.papers):
            raise ValueError("summary.papers ne correspond pas aux fiches")
        rows = [row for paper in self.papers for row in paper.evidence]
        for status in ("found", "not_reported", "missing", "human_review", "rejected", "llm_error"):
            if getattr(self.summary, status) != sum(row.status == status for row in rows):
                raise ValueError(f"summary.{status} incohérent")
        if self.summary.reused != sum(row.reused for row in rows):
            raise ValueError("summary.reused incohérent")
        return self
