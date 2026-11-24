"""Normalisation commune au matching de termes (agent A) et au contrôle d'ancrage (agent B)."""
import re
import unicodedata


def normalize_for_matching(text: str) -> str:
    """Minuscules, sans accents, ponctuation et tirets remplacés par des espaces, espaces simples."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^\w%]+", " ", stripped.casefold())
    return " ".join(cleaned.split())