"""Recherche lexicale (TF-IDF + cosinus) des chunks les plus proches d'une requête. Aucun réseau, aucun LLM.

Même méthode que `lexical_retrieve` du TD05. La requête doit être dans la langue des articles (anglais ici) :
les mots d'une requête française ne se retrouvent pas dans un texte anglais.
"""
import math
import re
from collections import Counter

from slrresearch.core.schemas.extraction import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", text.casefold())


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(left[word] * right[word] for word in set(left) & set(right))
    norm_left = math.sqrt(sum(v * v for v in left.values()))
    norm_right = math.sqrt(sum(v * v for v in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def lexical_retrieve(chunks: list[Chunk], query: str, k: int = 3) -> list[tuple[Chunk, float]]:
    """Les `k` chunks les plus proches (score décroissant, ordre du document à égalité). Un chunk de score 0 n'est jamais renvoyé."""
    if not chunks or k <= 0:
        return []
    documents = [tokenize(chunk.text) for chunk in chunks]
    n_docs = len(chunks)
    frequency = Counter(word for words in documents for word in set(words))

    def vector(words: list[str]) -> dict[str, float]:
        return {w: n * (math.log((1 + n_docs) / (1 + frequency.get(w, 0))) + 1) for w, n in Counter(words).items()}

    query_vector = vector(tokenize(query))
    scored = [(chunk, _cosine(query_vector, vector(words))) for chunk, words in zip(chunks, documents)]
    ranked = sorted(enumerate(scored), key=lambda item: (-item[1][1], item[0]))  # stable : égalité -> ordre du document
    return [pair for _, pair in ranked if pair[1] > 0][:k]