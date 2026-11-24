"""Validation d'un PDF téléchargé : est-ce bien un PDF, lisible, avec du texte, et le bon article ?

Aucun réseau ici : les fonctions reçoivent les octets. Pas d'OCR : un PDF sans couche texte
est refusé (verdict `no_text_layer`) et finira en revue humaine (porte G3).
"""
import io
import re
from dataclasses import dataclass

from slrresearch.screening.text import normalize_for_matching

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MIN_CHARS_PER_PAGE = 50.0   # moyenne en dessous de laquelle le PDF est jugé « scanné »
DEFAULT_MIN_TITLE_COVERAGE = 0.7    # part des mots significatifs du titre retrouvés dans le texte
DEFAULT_SAMPLE_PAGES = 10           # nombre de pages lues pour juger (le titre est cherché dans les 3 premières)
TITLE_PAGES = 3

# Mots trop courants pour prouver qu'on a le bon article.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "using", "that", "this", "are", "how", "can", "its",
    "les", "des", "une", "pour", "dans", "par", "sur", "avec", "est", "aux", "que", "qui",
}


@dataclass
class PdfCheck:
    ok: bool
    verdict: str  # ok | too_large | not_pdf | unreadable | no_text_layer | wrong_title
    pages: int = 0
    avg_chars: float = 0.0
    title_coverage: float = 0.0


def title_words(title: str) -> list[str]:
    """Mots significatifs d'un titre (normalisés, sans doublons, sans mots vides ni mots de moins de 3 lettres)."""
    seen, words = set(), []
    for word in normalize_for_matching(title).split():
        if len(word) >= 3 and word not in _STOPWORDS and word not in seen:
            seen.add(word)
            words.append(word)
    return words


def title_coverage(title: str, text: str) -> float:
    """Part (0..1) des mots significatifs du titre présents (mots entiers) dans `text`. Titre sans mot utile => 1.0."""
    words = title_words(title)
    if not words:
        return 1.0
    present = set(normalize_for_matching(text).split())
    return sum(1 for w in words if w in present) / len(words)


def looks_like_pdf(data: bytes) -> bool:
    return data.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"%PDF-")


def extract_pages(data: bytes, max_pages: int = DEFAULT_SAMPLE_PAGES, backend: str = "auto") -> tuple[list[str], int]:
    """Texte des `max_pages` premières pages et nombre total de pages. PyMuPDF d'abord, pypdf en repli.

    Lève ValueError si le PDF est illisible ou protégé par un mot de passe.
    """
    errors = []
    if backend in ("auto", "pymupdf"):
        try:
            return _extract_pymupdf(data, max_pages)
        except ImportError:
            errors.append("pymupdf absent")
        except Exception as exc:  # PyMuPDF lève plusieurs types selon la version
            errors.append(type(exc).__name__)
            if backend == "pymupdf":
                raise ValueError(f"PDF illisible ({type(exc).__name__})") from None
    if backend in ("auto", "pypdf"):
        try:
            return _extract_pypdf(data, max_pages)
        except ImportError:
            errors.append("pypdf absent")
        except Exception as exc:
            errors.append(type(exc).__name__)
    raise ValueError("PDF illisible (" + ", ".join(errors) + ")")


def _extract_pymupdf(data: bytes, max_pages: int) -> tuple[list[str], int]:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # anciennes versions de PyMuPDF
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("PDF protégé par mot de passe")
        total = doc.page_count
        return [(doc[i].get_text() or "").strip() for i in range(min(total, max_pages))], total


def _extract_pypdf(data: bytes, max_pages: int) -> tuple[list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("PDF protégé par mot de passe")
    total = len(reader.pages)
    return [(reader.pages[i].extract_text() or "").strip() for i in range(min(total, max_pages))], total


def validate_pdf(
    data: bytes,
    title: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    min_chars_per_page: float = DEFAULT_MIN_CHARS_PER_PAGE,
    min_title_coverage: float = DEFAULT_MIN_TITLE_COVERAGE,
    sample_pages: int = DEFAULT_SAMPLE_PAGES,
    backend: str = "auto",
) -> PdfCheck:
    """Contrôles dans l'ordre : taille, signature %PDF, lisibilité, couche texte, titre. Le premier échec décide."""
    if len(data) > max_bytes:
        return PdfCheck(False, "too_large")
    if not looks_like_pdf(data):
        return PdfCheck(False, "not_pdf")
    try:
        pages, total = extract_pages(data, sample_pages, backend)
    except ValueError:
        return PdfCheck(False, "unreadable")
    if not pages:
        return PdfCheck(False, "unreadable")
    avg = sum(len(p) for p in pages) / len(pages)
    if avg < min_chars_per_page:
        return PdfCheck(False, "no_text_layer", pages=total, avg_chars=avg)
    coverage = title_coverage(title, " ".join(pages[:TITLE_PAGES]))
    if coverage < min_title_coverage:
        return PdfCheck(False, "wrong_title", pages=total, avg_chars=avg, title_coverage=coverage)
    return PdfCheck(True, "ok", pages=total, avg_chars=avg, title_coverage=coverage)