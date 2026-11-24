# Pas de "from __future__ import annotations" ici (voir papers.py).
from typing import Literal

from pydantic import BaseModel, Field, model_validator

FieldStatus = Literal["verified", "not_reported", "no_context", "rejected", "invalid_format", "llm_error"]
# Seul `llm_error` (panne passagère : 429, réseau) est retenté à la reprise ; les autres statuts sont gardés.
RETRY_STATUSES = ("llm_error",)


class CardField(BaseModel):
    """Une information de la fiche. Seul un champ `verified` peut servir de fait : sa citation a été retrouvée
    mot pour mot dans le chunk cité. `not_reported` = le LLM dit que les extraits ne contiennent pas l'information."""

    field_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: FieldStatus
    value: str = ""
    chunk_id: str = ""
    page: int | None = Field(default=None, ge=1)
    quote: str = ""
    note: str = ""  # raison du statut ; obligatoire sauf pour verified et not_reported

    @model_validator(mode="after")
    def _coherent(self):
        if self.status == "verified":
            if not (self.value and self.chunk_id and self.quote and self.page):
                raise ValueError("un champ 'verified' doit avoir value, chunk_id, page et quote")
        elif self.status == "not_reported":
            if self.value or self.quote:
                raise ValueError("un champ 'not_reported' n'a ni valeur ni citation")
        elif not self.note:
            raise ValueError(f"un champ '{self.status}' doit expliquer la raison (note)")
        return self


class StudyCard(BaseModel):
    """Fichier `runs/<id>/study_cards/<paper>.json` : la fiche d'étude d'UN papier."""

    paper_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)  # texte des chunks + prompt + modèle + réglages : s'il change, la fiche est refaite
    model: str = Field(min_length=1)
    fields: list[CardField] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_fields(self):
        ids = [f.field_id for f in self.fields]
        if len(ids) != len(set(ids)):
            raise ValueError("field_id en double")
        return self


class CardEntry(BaseModel):
    """Une ligne de l'index : l'état de la fiche d'un papier extrait."""

    paper_id: str = Field(min_length=1)
    status: Literal["complete", "incomplete", "failed"]  # incomplete = au moins un champ en llm_error (à relancer)
    reason: str = ""                                      # obligatoire si failed
    file: str = ""                                        # obligatoire sinon
    counts: dict[str, int] = Field(default_factory=dict)  # nombre de champs par statut

    @model_validator(mode="after")
    def _coherent(self):
        if self.status == "failed" and not self.reason:
            raise ValueError("une fiche 'failed' doit expliquer la raison")
        if self.status != "failed" and not self.file:
            raise ValueError("une fiche complète ou incomplète doit avoir un fichier")
        return self


class StudyCardsLimits(BaseModel):
    """Les réglages utilisés pour CE run : à citer dans le rapport."""

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    pause: float = Field(ge=0)
    max_papers: int | None = Field(default=None, gt=0)
    fields: list[str] = Field(min_length=1)


class StudyCardsSummary(BaseModel):
    total: int = Field(ge=0)
    complete: int = Field(ge=0)
    incomplete: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped_not_extracted: int = Field(ge=0)  # papiers dont l'extraction (5a) a échoué : pas de fiche
    llm_calls: int = Field(ge=0)              # appels LLM faits pendant CE lancement (pas ceux d'un lancement précédent)
    tokens: int = Field(ge=0)
    fields: dict[str, int] = Field(default_factory=dict)  # champs par statut, tous papiers confondus

    @model_validator(mode="after")
    def _arithmetic(self):
        if self.complete + self.incomplete + self.failed != self.total:
            raise ValueError("complete + incomplete + failed doit valoir total")
        return self


class StudyCardsOutcome(BaseModel):
    """Artefact `study_cards` (étape 5b) : un enregistrement par papier extrait, dans l'ordre de `extraction`."""

    limits: StudyCardsLimits
    papers: list[CardEntry] = Field(min_length=1)
    summary: StudyCardsSummary

    @model_validator(mode="after")
    def _consistent(self):
        ids = [p.paper_id for p in self.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("papiers en double")
        if self.summary.total != len(self.papers):
            raise ValueError("summary.total ne correspond pas aux papiers")
        for status in ("complete", "incomplete", "failed"):
            if getattr(self.summary, status) != sum(1 for p in self.papers if p.status == status):
                raise ValueError(f"summary.{status} incohérent")
        return self