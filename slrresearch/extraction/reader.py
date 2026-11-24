"""Lecture d'UN PDF -> texte nettoyé par page -> chunks (`PaperText`), et fichier de sortie par papier.

Aucun réseau, aucun LLM, pas d'OCR : un PDF sans couche texte est refusé (`ExtractionError`).
"""
import json
import os
from pathlib import Path

from pydantic import ValidationError

from slrresearch.core.schemas.extraction import ExtractionParams, PaperText
from slrresearch.download import validate
from slrresearch.download.agent import safe_filename, sha256_hex
from slrresearch.extraction.text import chunk_pages, clean_page_text, find_repeated_lines

ALL_PAGES = 10**6  # `extract_pages` veut un nombre : « toutes » = un nombre très grand


class ExtractionError(Exception):
    """Le PDF ne peut pas être transformé en texte exploitable. `reason` est une phrase courte, affichée à l'utilisateur."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def extract_paper(paper_id: str, data: bytes, params: ExtractionParams, backend: str = "auto") -> PaperText:
    """Lit les pages (toutes, ou les `max_pages` premières), nettoie, découpe. Lève `ExtractionError`."""
    try:
        raw_pages, total = validate.extract_pages(data, params.max_pages or ALL_PAGES, backend)
    except ValueError as exc:
        raise ExtractionError(f"PDF illisible : {exc}") from None
    pages = [clean_page_text(p) for p in raw_pages]
    if not pages:
        raise ExtractionError("PDF sans page")
    chars = [len(p) for p in pages]
    if sum(chars) / len(pages) < params.min_chars_per_page:
        raise ExtractionError(f"pas de couche texte (moyenne {sum(chars) / len(pages):.0f} caractères par page) : sans OCR, inutilisable")
    chunks = chunk_pages(paper_id, pages, params.chunk_size, params.chunk_overlap)
    if not chunks:  # ne devrait pas arriver si la moyenne est suffisante, mais on ne laisse pas passer un papier vide
        raise ExtractionError("aucun texte après nettoyage")
    return PaperText(paper_id=paper_id, source_sha256=sha256_hex(data), params=params, total_pages=total,
                     pages_read=len(pages), page_chars=chars, repeated_lines=find_repeated_lines(pages), chunks=chunks)


# -- fichier par papier : runs/<id>/extraction/<paper>.json -----------------------------------------------------
DIRNAME = "extraction"


def paper_file(run_dir: Path, paper_id: str) -> Path:
    return run_dir / DIRNAME / (safe_filename(paper_id) + ".json")


def save_paper_text(path: Path, text: PaperText) -> None:
    """Écriture atomique (fichier temporaire puis os.replace) : jamais de JSON à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(text.model_dump(mode="json"), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(partial, path)


def load_paper_text(path: Path) -> PaperText | None:
    """None si le fichier est absent, illisible ou invalide (on refera alors l'extraction)."""
    try:
        return PaperText.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None