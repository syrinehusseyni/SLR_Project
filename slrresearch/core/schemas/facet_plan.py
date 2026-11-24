from typing import Literal

from pydantic import BaseModel, Field, model_validator


FacetCategory = Literal[
    "design", "metrics", "models", "data", "parameters", "results", "risks_limits", "methodology"
]


class FacetPlanSource(BaseModel):
    facet_id: str = Field(min_length=1)
    rq_id: str = Field(min_length=1)
    original_label: str = Field(min_length=1)
    original_description: str = Field(min_length=1)
    category: FacetCategory


class CanonicalFacet(BaseModel):
    id: FacetCategory
    label: str = Field(min_length=2)
    description: str = Field(min_length=5)
    query_en: str = Field(min_length=5)
    blocking: bool
    reuse_field: str = ""
    source_facet_ids: list[str] = Field(min_length=1)
    rq_ids: list[str] = Field(min_length=1)


class FacetPlanOutcome(BaseModel):
    version: str = Field(min_length=1)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    gate_id: str = Field(min_length=1)
    source_facets: list[FacetPlanSource] = Field(min_length=1)
    facets: list[CanonicalFacet] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _consistent(self):
        source_ids = [source.facet_id for source in self.source_facets]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("facet source IDs must be unique")
        mapped_ids = [facet_id for facet in self.facets for facet_id in facet.source_facet_ids]
        if sorted(source_ids) != sorted(mapped_ids):
            raise ValueError("each source facet must map to exactly one canonical facet")
        if len({facet.id for facet in self.facets}) != 8:
            raise ValueError("canonical facet categories must be unique")
        return self
