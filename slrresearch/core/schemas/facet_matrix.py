from typing import Literal

from pydantic import BaseModel, Field, model_validator

from slrresearch.core.schemas.facet_plan import FacetCategory


MatrixStatus = Literal["found", "not_reported", "missing", "contradictory"]
EvidenceStatus = Literal["found", "not_reported", "missing", "human_review", "rejected", "llm_error"]


class FacetMatrixCell(BaseModel):
    paper_id: str = Field(min_length=1)
    facet_id: str = Field(pattern=r"^F[0-9]+$")
    rq_id: str = Field(pattern=r"^RQ[0-9]+$")
    category: FacetCategory
    blocking: bool
    status: MatrixStatus
    evidence_status: EvidenceStatus
    value: str = ""
    quote: str = ""
    chunk_id: str = ""
    page: int | None = Field(default=None, ge=1)
    note: str = ""


class FacetMatrixPaper(BaseModel):
    paper_id: str = Field(min_length=1)
    cells: list[FacetMatrixCell] = Field(min_length=1)


class FacetMatrixSummary(BaseModel):
    papers: int = Field(ge=0)
    cells: int = Field(ge=0)
    found: int = Field(ge=0)
    not_reported: int = Field(ge=0)
    missing: int = Field(ge=0)
    contradictory: int = Field(ge=0)
    blocking_missing: int = Field(ge=0)


class FacetMatrixOutcome(BaseModel):
    version: str = Field(min_length=1)
    plan_hash: str = Field(min_length=64, max_length=64)
    papers: list[FacetMatrixPaper] = Field(min_length=1)
    summary: FacetMatrixSummary

    @model_validator(mode="after")
    def _consistent(self):
        rows = [cell for paper in self.papers for cell in paper.cells]
        if self.summary.papers != len(self.papers) or self.summary.cells != len(rows):
            raise ValueError("résumé de matrice incohérent")
        for status in ("found", "not_reported", "missing", "contradictory"):
            if getattr(self.summary, status) != sum(cell.status == status for cell in rows):
                raise ValueError(f"summary.{status} incohérent")
        if self.summary.blocking_missing != sum(cell.blocking and cell.status == "missing" for cell in rows):
            raise ValueError("summary.blocking_missing incohérent")
        return self
