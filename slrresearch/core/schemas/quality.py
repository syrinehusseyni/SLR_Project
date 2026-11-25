from typing import Literal

from pydantic import BaseModel, Field, model_validator


QualityRating = Literal["yes", "partial", "no", "not_reported"]
QualityItemStatus = Literal["accepted", "human_review", "rejected", "llm_error"]
QualityPaperStatus = Literal["complete", "waiting_human", "incomplete", "rejected"]


class QualityAssessment(BaseModel):
    criterion_id: str = Field(min_length=1)
    rating: QualityRating | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""
    quote: str = ""
    chunk_id: str = ""
    page: int | None = Field(default=None, ge=1)
    quote_in_chunk: bool = False
    status: QualityItemStatus
    note: str = ""
    review_decision: Literal["approved", "rejected"] | None = None
    reviewer: str = ""


class QualityPaper(BaseModel):
    paper_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=64, max_length=64)
    status: QualityPaperStatus
    review_item_id: str = ""
    assessments: list[QualityAssessment] = Field(min_length=1)
    note: str = ""


class QualitySummary(BaseModel):
    papers: int = Field(ge=0)
    complete: int = Field(ge=0)
    waiting_human: int = Field(ge=0)
    incomplete: int = Field(ge=0)
    rejected: int = Field(ge=0)
    assessments: int = Field(ge=0)
    accepted: int = Field(ge=0)
    human_review: int = Field(ge=0)
    rejected_items: int = Field(ge=0)
    llm_errors: int = Field(ge=0)
    ratings: dict[str, int]
    llm_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)


class QualityOutcome(BaseModel):
    version: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    plan_hash: str = Field(min_length=64, max_length=64)
    papers: list[QualityPaper] = Field(min_length=1)
    summary: QualitySummary

    @model_validator(mode="after")
    def _consistent(self):
        if self.summary.papers != len(self.papers):
            raise ValueError("summary.papers ne correspond pas aux fiches qualité")
        for status in ("complete", "waiting_human", "incomplete", "rejected"):
            if getattr(self.summary, status) != sum(p.status == status for p in self.papers):
                raise ValueError(f"summary.{status} incohérent")
        items = [item for paper in self.papers for item in paper.assessments]
        if self.summary.assessments != len(items):
            raise ValueError("summary.assessments incohérent")
        for status in ("accepted", "human_review", "rejected", "llm_error"):
            field = "rejected_items" if status == "rejected" else "llm_errors" if status == "llm_error" else status
            if getattr(self.summary, field) != sum(item.status == status for item in items):
                raise ValueError(f"summary.{field} incohérent")
        expected_ratings = {rating: sum(item.rating == rating for item in items) for rating in
                            ("yes", "partial", "no", "not_reported")}
        if self.summary.ratings != expected_ratings:
            raise ValueError("summary.ratings incohérent")
        return self
