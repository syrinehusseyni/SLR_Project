"""Nettoyage minimal du texte d'une page et découpage en chunks. Aucun réseau, aucun LLM.

Principe : tout ce que l'étape 6 citera « mot pour mot » doit se retrouver dans un chunk. On ne corrige donc que
ce qui est sûr (ligatures, caractères de contrôle, mots coupés en fin de ligne) ; en-têtes et pieds de page sont
SIGNALÉS (`find_repeated_lines`), pas supprimés. Tableaux, formules et colonnes ne sont pas reconstruits.
"""
import re
from collections import Counter

from slrresearch.core.schemas.extraction import Chunk

CLEANING_VERSION = "1"

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_EOL = re.compile(r"(?<=[A-Za-zÀ-ÿ])-[ \t]*\n[ \t]*(?=[a-zà-ÿ])")  # « classi-\nfication » -> « classification »
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_page_text(text: str) -> str:
    """Nettoyage minimal et déterministe du texte d'UNE page.

    Limite connue : « self-\\nsupervised » devient « selfsupervised » (un tiret de fin de ligne est ambigu).
    """
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = text.replace("\xad", "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_EOL.sub("", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _line_pattern(line: str) -> str:
    return re.sub(r"[0-9]+", "#", " ".join(line.split()))  # « Page 12 » et « Page 13 » sont le même motif


def find_repeated_lines(pages: list[str], min_pages: int = 3, min_share: float = 0.4, max_lines: int = 10) -> list[str]:
    """Lignes (courtes) qui reviennent sur au moins `min_share` des pages : en-têtes et pieds de page probables."""
    if len(pages) < min_pages:
        return []
    seen: Counter = Counter()
    example: dict[str, str] = {}
    for page in pages:
        for pattern in {_line_pattern(line) for line in page.split("\n") if 4 <= len(line.strip()) <= 120}:
            seen[pattern] += 1
            example.setdefault(pattern, pattern)
    needed = max(min_pages, min_share * len(pages))
    return [example[p] for p, n in seen.most_common() if n >= needed][:max_lines]


def validate_chunking(size: int, overlap: int) -> None:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Il faut size > 0 et 0 <= overlap < size")


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Fenêtres d'au plus `size` caractères, coupées sur une espace (jamais au milieu d'un mot, sauf mot géant)."""
    pieces: list[str] = []
    length = len(text)
    start = 0
    while start < length:
        end = min(start + size, length)
        if end < length:
            cut = max(text.rfind(" ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        nxt = max(end - overlap, start + 1)
        while 0 < nxt < length and not text[nxt - 1].isspace():  # ne pas démarrer au milieu d'un mot
            nxt += 1
            if nxt >= end:
                nxt = end
                break
        start = nxt
    return pieces


def chunk_pages(paper_id: str, pages: list[str], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Découpe des pages (page 1 = pages[0]) en chunks `{paper_id}_p{page}_c{n}` ; une page vide ne produit rien."""
    validate_chunking(size, overlap)
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, 1):
        for number, piece in enumerate(_split(page_text, size, overlap), 1):
            chunks.append(Chunk(paper_id=paper_id, chunk_id=f"{paper_id}_p{page_number}_c{number}", page=page_number, text=piece))
    return chunks