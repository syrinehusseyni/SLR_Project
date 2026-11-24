# ATTENTION : pas de "from __future__ import annotations" dans les schemas
# (il casse la resolution de Literal avec Pydantic quand le module est charge par importlib).
from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    """Un article trouve par une source. Etend le PaperRecord du TD03."""

    id: str = Field(min_length=1)  # stable : "W123..." (OpenAlex), sinon derive du DOI
    title: str = Field(min_length=1)
    abstract: str = ""
    doi: str = ""
    source: str = Field(min_length=1)  # openalex | crossref | semantic_scholar | arxiv
    url: str = ""
    open_access: bool = False
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    pdf_url: str = ""  # indispensable pour l'etape 4 (acquisition)

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, v: str) -> str:
        # Un DOI se compare en minuscules, sans prefixe d'URL.
        v = v.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
            if v.startswith(prefix):
                v = v[len(prefix):]
        return v

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("titre vide")
        return v


class SearchResult(BaseModel):
    """Resultat d'une recherche : c'est CET objet qu'on ecrit comme artefact."""

    query: str
    source: str
    papers: list[Paper] = Field(default_factory=list)
    error: str = ""  # ex: "rate_limited" ; vide si tout va bien