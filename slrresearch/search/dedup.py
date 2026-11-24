"""Dédoublonnage : même DOI, même identifiant, ou même titre normalisé => une seule notice.

Les champs manquants sont complétés depuis les doublons (jamais inventés). L'identifiant final est stable :
OpenAlex (W...) si connu, sinon dérivé du DOI, sinon celui de la notice retenue.
"""
import re
import unicodedata
from dataclasses import dataclass

from slrresearch.core.schemas.papers import Paper
from slrresearch.search.sources import slug

PRIORITY = {"openalex": 0, "semanticscholar": 1, "crossref": 2, "arxiv": 3}
MIN_TITLE_CHARS = 20  # en dessous, un titre est trop générique pour servir de clé


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


@dataclass
class DedupResult:
    papers: list[Paper]
    found_in: dict[str, list[str]]
    duplicates_removed: int


def _keys(paper: Paper) -> list[tuple[str, str]]:
    keys = [("id", paper.id)]
    if paper.doi:
        keys.append(("doi", paper.doi))
    title = normalize_title(paper.title)
    if len(title) >= MIN_TITLE_CHARS:
        keys.append(("title", title))
    return keys


def _merge(group: list[Paper]) -> Paper:
    ordered = sorted(enumerate(group), key=lambda ip: (PRIORITY.get(ip[1].source, 9), ip[0]))
    members = [p for _, p in ordered]
    rep = members[0]

    def first(attr):
        return next((getattr(p, attr) for p in members if getattr(p, attr)), getattr(rep, attr))

    doi = first("doi")
    openalex_id = next((p.id for p in members if p.source == "openalex" and re.fullmatch(r"W\d+", p.id)), "")
    paper_id = openalex_id or (f"doi-{slug(doi)}" if doi else rep.id)
    return rep.model_copy(update={
        "id": paper_id,
        "abstract": max((p.abstract for p in members), key=len),
        "doi": doi,
        "url": first("url"),
        "year": first("year"),
        "authors": first("authors"),
        "venue": first("venue"),
        "pdf_url": first("pdf_url"),
        "open_access": any(p.open_access for p in members),
    })


def deduplicate(records: list[Paper]) -> DedupResult:
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, paper in enumerate(records):
        for key in _keys(paper):
            if key in seen:
                a, b = find(i), find(seen[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)

    papers: list[Paper] = []
    found_in: dict[str, list[str]] = {}
    for root in sorted(groups):
        members = [records[i] for i in groups[root]]
        merged = _merge(members)
        papers.append(merged)
        found_in[merged.id] = sorted({p.source for p in members})
    return DedupResult(papers, found_in, len(records) - len(papers))