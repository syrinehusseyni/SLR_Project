# Pas de "from __future__ import annotations" ici (voir papers.py).
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DownloadAttempt(BaseModel):
    """Une tentative de telechargement, tracee meme si elle echoue."""

    resolver: str  # ex: openalex_oa | unpaywall | s2_open_access | arxiv | crossref | manual
    url: str
    http_status: int | None = None
    validation: str = ""  # ex: "ok", "not_pdf", "wrong_title", "no_text_layer"
    ok: bool = False


class DocumentRecord(BaseModel):
    """Etat d'acquisition du PDF d'un article. Sans OCR : pas de texte => manual_required."""

    paper_id: str = Field(min_length=1)
    status: Literal["downloaded", "manual_required", "unavailable"]
    attempts: list[DownloadAttempt] = Field(default_factory=list)
    pdf_path: str = ""
    sha256: str = ""
    has_text_layer: bool = False

    @model_validator(mode="after")
    def _downloaded_needs_file(self):
        if self.status == "downloaded" and not (self.pdf_path and self.sha256):
            raise ValueError("un document 'downloaded' doit avoir pdf_path et sha256")
        return self


class DownloadLimits(BaseModel):
    """Les limites utilisees pour CE run : a citer dans le rapport (choix de cout assume, pas une contrainte du pipeline)."""

    max_papers: int | None = None
    on_missing: Literal["review", "unavailable"] = "review"  # que faire d'un papier sans PDF en accès libre
    max_mb: float = Field(gt=0)
    max_http_calls: int = Field(ge=0)
    pause: float = Field(ge=0)
    min_title_coverage: float = Field(ge=0, le=1)
    min_chars_per_page: float = Field(ge=0)


class DocumentsProgress(BaseModel):
    """Artefact de travail `documents_progress` : l'etat de chaque papier deja traite (reprise apres arret)."""

    records: dict[str, DocumentRecord] = Field(default_factory=dict)


class DocumentsSummary(BaseModel):
    total: int = Field(ge=0)
    downloaded: int = Field(ge=0)
    manual_required: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    by_resolver: dict[str, int] = Field(default_factory=dict)  # resolveur qui a fourni le PDF valide

    @model_validator(mode="after")
    def _arithmetic(self):
        if self.downloaded + self.manual_required + self.unavailable != self.total:
            raise ValueError("downloaded + manual_required + unavailable doit valoir total")
        if sum(self.by_resolver.values()) != self.downloaded:
            raise ValueError("by_resolver doit sommer au nombre de documents telecharges")
        return self


class DocumentsOutcome(BaseModel):
    """Artefact `documents` (etape 4) : un enregistrement par papier inclus a l'etape 3, dans l'ordre du screening."""

    limits: DownloadLimits
    records: list[DocumentRecord] = Field(min_length=1)
    summary: DocumentsSummary

    @model_validator(mode="after")
    def _consistent(self):
        ids = [r.paper_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("papiers en double")
        if self.summary.total != len(self.records):
            raise ValueError("summary.total ne correspond pas aux enregistrements")
        for status in ("downloaded", "manual_required", "unavailable"):
            if getattr(self.summary, status) != sum(1 for r in self.records if r.status == status):
                raise ValueError(f"summary.{status} incoherent")
        return self