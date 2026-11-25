"""Project-specific quality checklist and strict response parsing."""
import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field


RUBRIC_VERSION = "slr-screening-quality-v1"
PROMPT_NAME = "quality_appraisal.txt"


@dataclass(frozen=True)
class Criterion:
    id: str
    label: str
    question: str
    source_categories: tuple[str, ...]


CHECKLIST = (
    Criterion("Q1", "Research objective", "Is the study objective or research question clearly stated?", ("design", "methodology")),
    Criterion("Q2", "Dataset and sample", "Are the dataset, sample, or study population described clearly enough to understand what was evaluated?", ("data",)),
    Criterion("Q3", "Comparator or baseline", "Where a comparison is made, is the comparator or baseline identified clearly? Do not penalize a study merely because it is not comparative.", ("design", "models", "results")),
    Criterion("Q4", "Evaluation measures", "Are the evaluation outcomes or metrics identified clearly enough to interpret the reported evaluation?", ("metrics", "results")),
    Criterion("Q5", "Reproducibility detail", "Does the available evidence describe important model, workflow, data, or configuration details needed to understand or reproduce the approach?", ("models", "parameters", "methodology", "data")),
    Criterion("Q6", "Limitations and validity", "Does the available evidence discuss limitations, bias, uncertainty, or threats that could affect the findings?", ("risks_limits", "results", "methodology")),
)
CRITERION_IDS = tuple(item.id for item in CHECKLIST)


class ProposedAssessment(BaseModel):
    criterion_id: str = Field(min_length=1)
    rating: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    quote: str = ""
    chunk_id: str = ""


class ProposedQualityResponse(BaseModel):
    assessments: list[ProposedAssessment] = Field(min_length=len(CHECKLIST), max_length=len(CHECKLIST))


def fingerprint(paper_id: str, source_sha256: str, evidence_rows: list[dict], model: str,
                prompt: str, review_confidence: float) -> str:
    material = json.dumps([RUBRIC_VERSION, paper_id, source_sha256, evidence_rows, model, prompt,
                           [item.__dict__ for item in CHECKLIST], review_confidence],
                          sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def parse_response(text: str) -> ProposedQualityResponse:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("aucun objet JSON dans la réponse qualité")
    response = ProposedQualityResponse.model_validate_json(text[start:end + 1])
    ids = [item.criterion_id for item in response.assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(CRITERION_IDS):
        raise ValueError("la réponse doit contenir exactement une évaluation pour Q1 à Q6")
    allowed = {"yes", "partial", "no", "not_reported"}
    for item in response.assessments:
        if item.rating not in allowed:
            raise ValueError(f"rating invalide pour {item.criterion_id}: {item.rating}")
        if item.rating == "not_reported" and (item.quote or item.chunk_id):
            raise ValueError(f"not_reported ne doit pas inventer de citation ({item.criterion_id})")
        if item.rating != "not_reported" and (not item.quote or not item.chunk_id):
            raise ValueError(f"citation obligatoire pour {item.criterion_id}")
    return response


def build_prompt(template: str, context: dict) -> str:
    return template.replace("{{context}}", json.dumps(context, ensure_ascii=False, indent=2))
