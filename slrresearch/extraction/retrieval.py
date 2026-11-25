"""Recherche lexicale et sémantique des chunks, avec fusion hybride par RRF.

La voie lexicale est locale et déterministe. La voie dense utilise le endpoint local Ollama `/api/embed`.
La fusion RRF combine les deux classements sans comparer leurs scores de nature différente.
"""
import json
import hashlib
import math
import re
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from slrresearch.core.schemas.extraction import Chunk
from slrresearch.core.errors import ConfigError


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


class OllamaEmbeddingClient:
    """Petit client du endpoint local Ollama; n'exige aucun changement de fournisseur LLM."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text",
                 timeout: float = 120.0, batch_size: int = 64):
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError("OLLAMA_BASE_URL doit commencer par http:// ou https://")
        if not model.strip() or timeout <= 0 or batch_size <= 0:
            raise ConfigError("modèle, timeout et taille de lot des embeddings doivent être valides")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        self.batch_size = batch_size

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": inputs}, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + "/api/embed", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ConfigError(f"Ollama embeddings a répondu HTTP {exc.code}; vérifiez le modèle {self.model!r}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ConfigError(f"Impossible de joindre Ollama embeddings à {self.base_url}; vérifiez Ollama et OLLAMA_BASE_URL.") from exc
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ConfigError("réponse Ollama embeddings invalide ou nombre de vecteurs inattendu")
        if any(not isinstance(v, list) or not v or any(not isinstance(x, (int, float)) for x in v) for v in vectors):
            raise ConfigError("Ollama embeddings a renvoyé un vecteur vide ou invalide")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            result.extend(self._embed(texts[offset:offset + self.batch_size]))
        return result

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class HybridRetriever:
    """Lexical, semantic, or reciprocal-rank fusion retrieval with a per-paper vector cache."""

    MODES = ("hybrid", "lexical", "semantic")

    def __init__(self, embeddings, mode: str = "hybrid", rrf_k: int = 60, candidate_k: int = 20):
        if mode not in self.MODES:
            raise ValueError(f"mode de recherche inconnu : {mode!r}; valeurs possibles : {self.MODES}")
        if rrf_k <= 0 or candidate_k <= 0:
            raise ValueError("rrf_k et candidate_k doivent être > 0")
        self.embeddings = embeddings
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._vectors: dict[str, tuple[str, list[list[float]]]] = {}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ConfigError("dimension d'embedding incohérente entre requête et chunks")
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0

    def _semantic(self, chunks: list[Chunk], query: str, k: int) -> list[tuple[Chunk, float]]:
        if not chunks or k <= 0:
            return []
        paper_id = chunks[0].paper_id
        digest = hashlib.sha256("\n".join(f"{c.chunk_id}\n{c.text}" for c in chunks).encode("utf-8")).hexdigest()
        signature = digest
        cached = self._vectors.get(paper_id)
        if cached is None or cached[0] != signature:
            vectors = self.embeddings.embed_documents([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise ConfigError("le modèle embeddings n'a pas renvoyé un vecteur par chunk")
            self._vectors[paper_id] = (signature, vectors)
        else:
            vectors = cached[1]
        query_vector = self.embeddings.embed_query(query)
        scored = [(chunk, self._cosine(query_vector, vector)) for chunk, vector in zip(chunks, vectors)]
        ranked = sorted(enumerate(scored), key=lambda item: (-item[1][1], item[0]))
        return [pair for _, pair in ranked[:k]]

    def retrieve(self, chunks: list[Chunk], query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not chunks or k <= 0:
            return []
        if self.mode == "lexical":
            return lexical_retrieve(chunks, query, k)
        candidate_k = min(len(chunks), max(self.candidate_k, k))
        semantic = self._semantic(chunks, query, candidate_k)
        if self.mode == "semantic":
            return semantic[:k]
        lexical = lexical_retrieve(chunks, query, candidate_k)
        order = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        scores: dict[str, float] = {}
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for ranking in (lexical, semantic):
            for rank, (chunk, _) in enumerate(ranking, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))[:k]
        return [(by_id[cid], scores[cid]) for cid in ranked_ids]
