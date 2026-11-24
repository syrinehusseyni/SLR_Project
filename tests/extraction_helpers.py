"""Aides pour tester l'étape 5a : de vrais petits PDF (PyMuPDF) et un run avec l'artefact `documents` prêt."""
import pymupdf

from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.documents import (
    DocumentRecord, DocumentsOutcome, DocumentsSummary, DownloadAttempt, DownloadLimits,
)
from slrresearch.download.agent import sha256_hex, store_pdf


def make_pdf(pages: list[str]) -> bytes:
    """PDF avec une page par texte ; "" = page sans aucun texte (comme une image scannée)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def prose(tag: str, sentences: int = 12) -> str:
    """Texte reconnaissable (les phrases portent le tag) et assez long pour passer le seuil de caractères par page."""
    return " ".join(f"Sentence {i} of {tag} reports recall {i / 10:.1f} on the evaluation set." for i in range(1, sentences + 1))


def setup_documents(store, specs: dict[str, bytes | str | None]):
    """specs : paper_id -> octets d'un PDF (downloaded), 'unavailable', 'manual_required', ou None (downloaded mais fichier absent)."""
    records = []
    for paper_id, spec in specs.items():
        if isinstance(spec, bytes):
            path, digest = store_pdf(paper_id, spec, store.run_dir / "documents")
            records.append(DocumentRecord(paper_id=paper_id, status="downloaded", pdf_path=path, sha256=digest, has_text_layer=True,
                                          attempts=[DownloadAttempt(resolver="arxiv", url="u", ok=True, validation="ok")]))
        elif spec is None:
            records.append(DocumentRecord(paper_id=paper_id, status="downloaded", pdf_path=f"documents/{paper_id}.pdf",
                                          sha256=sha256_hex(b"absent")))
        else:
            records.append(DocumentRecord(paper_id=paper_id, status=spec))
    count = lambda s: sum(1 for r in records if r.status == s)  # noqa: E731
    by = {"arxiv": count("downloaded")} if count("downloaded") else {}
    store.write_artifact("documents", DocumentsOutcome, DocumentsOutcome(
        limits=DownloadLimits(max_mb=50, max_http_calls=400, pause=1, min_title_coverage=0.7, min_chars_per_page=50),
        records=records,
        summary=DocumentsSummary(total=len(records), downloaded=count("downloaded"), manual_required=count("manual_required"),
                                 unavailable=count("unavailable"), by_resolver=by),
    ).model_dump())
    return build_context(store, Settings.from_env({}), make_model=lambda s, r: None)