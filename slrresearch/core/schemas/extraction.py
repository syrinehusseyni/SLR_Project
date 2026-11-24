# Pas de "from __future__ import annotations" ici (voir papers.py).
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_CHUNK_ID = re.compile(r"^(?P<paper>.+)_p(?P<page>[0-9]+)_c(?P<n>[0-9]+)$")


class Chunk(BaseModel):
    """Un fragment de texte avec sa provenance. `chunk_id` = "{paper_id}_p{page}_c{n}" (stable, jamais recréé)."""

    paper_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _id_matches(self):
        found = _CHUNK_ID.match(self.chunk_id)
        if not found or found["paper"] != self.paper_id or int(found["page"]) != self.page:
            raise ValueError(f"chunk_id {self.chunk_id!r} incohérent avec paper_id={self.paper_id!r} et page={self.page}")
        return self


class ExtractionParams(BaseModel):
    """Réglages qui déterminent le texte produit : s'ils changent, un papier déjà extrait doit être refait."""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    max_pages: int | None = Field(default=None, gt=0)  # None = toutes les pages
    min_chars_per_page: float = Field(ge=0)
    cleaning: str = Field(min_length=1)  # version des règles de nettoyage

    @model_validator(mode="after")
    def _overlap_below_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("il faut chunk_overlap < chunk_size")
        return self


class PaperText(BaseModel):
    """Fichier `runs/<id>/extraction/<paper>.json` : le texte d'UN papier, découpé en chunks."""

    paper_id: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)  # empreinte du PDF lu (détecte un PDF remplacé)
    params: ExtractionParams
    total_pages: int = Field(ge=1)
    pages_read: int = Field(ge=1)
    page_chars: list[int]  # caractères par page lue (0 = page sans texte, ex. image)
    repeated_lines: list[str] = Field(default_factory=list)  # en-têtes / pieds de page probables (signalés, pas supprimés)
    chunks: list[Chunk] = Field(min_length=1)

    @property
    def truncated(self) -> bool:
        return self.pages_read < self.total_pages

    @model_validator(mode="after")
    def _consistent(self):
        if self.pages_read > self.total_pages:
            raise ValueError("pages_read dépasse total_pages")
        if len(self.page_chars) != self.pages_read:
            raise ValueError("page_chars doit avoir une valeur par page lue")
        ids = [c.chunk_id for c in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk_id en double")
        for chunk in self.chunks:
            if chunk.paper_id != self.paper_id:
                raise ValueError(f"chunk {chunk.chunk_id} appartient à un autre papier")
            if chunk.page > self.pages_read:
                raise ValueError(f"chunk {chunk.chunk_id} : page au-delà des pages lues")
        return self


class ExtractedPaperEntry(BaseModel):
    """Une ligne de l'index : l'état d'extraction d'un papier téléchargé."""

    paper_id: str = Field(min_length=1)
    status: Literal["extracted", "failed"]
    reason: str = ""            # obligatoire si failed
    file: str = ""              # chemin relatif au dossier du run, obligatoire si extracted
    source_sha256: str = ""
    total_pages: int = Field(default=0, ge=0)
    pages_read: int = Field(default=0, ge=0)
    chars: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    empty_pages: int = Field(default=0, ge=0)
    repeated_lines: int = Field(default=0, ge=0)
    truncated: bool = False

    @model_validator(mode="after")
    def _coherent(self):
        if self.status == "extracted" and not (self.file and self.source_sha256 and self.chunks):
            raise ValueError("un papier 'extracted' doit avoir file, source_sha256 et au moins un chunk")
        if self.status == "failed" and not self.reason:
            raise ValueError("un papier 'failed' doit expliquer la raison")
        return self


class ExtractionLimits(ExtractionParams):
    """Les réglages utilisés pour CE run : à citer dans le rapport."""

    max_papers: int | None = Field(default=None, gt=0)


class ExtractionSummary(BaseModel):
    total: int = Field(ge=0)
    extracted: int = Field(ge=0)
    failed: int = Field(ge=0)
    chunks: int = Field(ge=0)
    skipped_unavailable: int = Field(ge=0)  # papiers inclus mais sans PDF (étape 4) : non extraits

    @model_validator(mode="after")
    def _arithmetic(self):
        if self.extracted + self.failed != self.total:
            raise ValueError("extracted + failed doit valoir total")
        return self


class ExtractionOutcome(BaseModel):
    """Artefact `extraction` (étape 5a) : un enregistrement par papier téléchargé, dans l'ordre de `documents`."""

    limits: ExtractionLimits
    papers: list[ExtractedPaperEntry] = Field(min_length=1)
    summary: ExtractionSummary

    @model_validator(mode="after")
    def _consistent(self):
        ids = [p.paper_id for p in self.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("papiers en double")
        if self.summary.total != len(self.papers):
            raise ValueError("summary.total ne correspond pas aux papiers")
        for status in ("extracted", "failed"):
            if getattr(self.summary, status) != sum(1 for p in self.papers if p.status == status):
                raise ValueError(f"summary.{status} incohérent")
        if self.summary.chunks != sum(p.chunks for p in self.papers):
            raise ValueError("summary.chunks incohérent")
        return self