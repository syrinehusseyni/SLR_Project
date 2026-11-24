"""Agent de téléchargement d'UN papier : essaie les résolveurs légaux dans l'ordre, valide chaque PDF.

Le programme contrôle tout (pas de LLM ici). Chaque tentative est tracée dans `DocumentRecord.attempts`.
Résultat : `downloaded` (PDF valide enregistré) ou `manual_required` (aucun PDF valide : paywall, mauvais PDF,
PDF scanné, aucune adresse) => porte G3. `unavailable` n'est jamais décidé par la machine : c'est l'humain, en G3.
"""
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from slrresearch.core.schemas.documents import DocumentRecord, DownloadAttempt
from slrresearch.core.schemas.papers import Paper
from slrresearch.download import validate
from slrresearch.download.resolvers import RESOLVERS, user_agent
from slrresearch.search.http import HttpClient, SearchError
from slrresearch.search.sources import SearchConfig


@dataclass
class DownloadSettings:
    """Tous les réglages sont des paramètres d'exécution (option de commande, demain champ d'interface)."""

    max_bytes: int = validate.DEFAULT_MAX_BYTES
    min_chars_per_page: float = validate.DEFAULT_MIN_CHARS_PER_PAGE
    min_title_coverage: float = validate.DEFAULT_MIN_TITLE_COVERAGE
    sample_pages: int = validate.DEFAULT_SAMPLE_PAGES
    max_urls_per_resolver: int = 3


def safe_filename(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._") or "paper"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _limiter_source(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return "arxiv" if host == "arxiv.org" or host.endswith(".arxiv.org") else "pdf"


def check_bytes(data: bytes, title: str, settings: DownloadSettings) -> validate.PdfCheck:
    return validate.validate_pdf(
        data, title,
        max_bytes=settings.max_bytes,
        min_chars_per_page=settings.min_chars_per_page,
        min_title_coverage=settings.min_title_coverage,
        sample_pages=settings.sample_pages,
    )


def store_pdf(paper_id: str, data: bytes, documents_dir: Path) -> tuple[str, str]:
    """Écrit le PDF (fichier temporaire puis remplacement) et renvoie (chemin relatif au dossier du run, sha256)."""
    documents_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(paper_id) + ".pdf"
    target = documents_dir / name
    partial = documents_dir / (name + ".part")
    partial.write_bytes(data)
    os.replace(partial, target)
    return f"{documents_dir.name}/{name}", sha256_hex(data)


def stored_file_ok(record: DocumentRecord, run_dir: Path) -> bool:
    """Reprise : le PDF d'un enregistrement `downloaded` existe-t-il encore, intact ?"""
    if record.status != "downloaded" or not record.pdf_path:
        return False
    path = run_dir / record.pdf_path
    try:
        return path.is_file() and sha256_hex(path.read_bytes()) == record.sha256
    except OSError:
        return False


def _fetch(http: HttpClient, config: SearchConfig, paper: Paper, resolver: str, url: str,
           settings: DownloadSettings) -> tuple[DownloadAttempt, bytes | None]:
    headers = {"User-Agent": user_agent(config), "Accept": "application/pdf,*/*;q=0.8"}
    try:
        response = http.get(_limiter_source(url), url, None, headers)  # BudgetExceeded remonte
    except SearchError as exc:
        return DownloadAttempt(resolver=resolver, url=url, http_status=exc.status, validation="http_error"), None
    except (httpx.HTTPError, httpx.InvalidURL) as exc:  # ex. trop de redirections ; jamais str(exc)
        return DownloadAttempt(resolver=resolver, url=url, validation=f"network_error:{type(exc).__name__}"), None
    check = check_bytes(response.content, paper.title, settings)
    attempt = DownloadAttempt(resolver=resolver, url=url, http_status=response.status_code,
                              validation=check.verdict, ok=check.ok)
    return attempt, (response.content if check.ok else None)


def download_paper(paper: Paper, http: HttpClient, config: SearchConfig, settings: DownloadSettings,
                   documents_dir: Path) -> DocumentRecord:
    """Essaie les résolveurs dans l'ordre et s'arrête au premier PDF valide (les suivants ne sont pas interrogés)."""
    attempts: list[DownloadAttempt] = []
    tried: set[str] = set()
    for resolve in RESOLVERS:
        resolution = resolve(paper, http, config)
        if resolution.note is not None:
            attempts.append(resolution.note)
        for url in resolution.urls[: settings.max_urls_per_resolver]:
            if url in tried:
                continue
            tried.add(url)
            attempt, data = _fetch(http, config, paper, resolution.resolver, url, settings)
            attempts.append(attempt)
            if data is not None:
                pdf_path, digest = store_pdf(paper.id, data, documents_dir)
                return DocumentRecord(paper_id=paper.id, status="downloaded", attempts=attempts,
                                      pdf_path=pdf_path, sha256=digest, has_text_layer=True)
    return DocumentRecord(paper_id=paper.id, status="manual_required", attempts=attempts)