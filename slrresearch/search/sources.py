"""Clients des 4 sources : construction de la requête HTTP + lecture (parsing) de la réponse.

Les fonctions `parse_*` sont pures (aucun réseau) : elles transforment la réponse brute en `Paper`.
Une notice inutilisable (sans titre, refusée par le schéma) est comptée dans `skipped`, jamais inventée.
Formats de réponse vérifiés dans les documentations le 14 nov. 2026 (OpenAlex) ; Crossref, Semantic Scholar
et arXiv d'après leurs documentations habituelles : à confirmer par le test `live`.
"""
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from pydantic import ValidationError

from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import SourceQuery
from slrresearch.search.http import HttpClient, MissingCredential, SearchError

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"

OPENALEX_SELECT = "id,doi,title,publication_year,abstract_inverted_index,authorships,primary_location,best_oa_location,open_access"
CROSSREF_SELECT = "DOI,title,abstract,issued,author,container-title,URL,type"
# Crossref mélange articles, comptes rendus d'évaluation (« peer-review »), jeux de données, subventions...
# On ne garde que ce qui peut être une étude. Un type absent n'est pas un motif de rejet.
CROSSREF_KEEP_TYPES = {"journal-article", "proceedings-article", "posted-content", "book-chapter",
                       "report", "dissertation", "monograph", "book"}
SEMANTIC_FIELDS = "paperId,title,abstract,year,authors,venue,externalIds,openAccessPdf,isOpenAccess,url"

_ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


@dataclass
class SearchConfig:
    openalex_api_key: str = ""
    semantic_scholar_api_key: str = ""
    contact_email: str = ""

    @classmethod
    def from_env(cls, mapping) -> "SearchConfig":
        return cls(
            openalex_api_key=(mapping.get("OPENALEX_API_KEY") or "").strip(),
            semantic_scholar_api_key=(mapping.get("SEMANTIC_SCHOLAR_API_KEY") or "").strip(),
            contact_email=(mapping.get("CONTACT_EMAIL") or "").strip(),
        )


@dataclass
class ParseResult:
    papers: list[Paper]
    skipped: int = 0


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def _year(value) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1800 <= year <= 2100 else None


def _text(value) -> str:
    return " ".join(str(value).split()) if value else ""


def _build(**fields) -> Paper | None:
    try:
        return Paper(**fields)
    except ValidationError:
        return None


# ---------------------------------------------------------------- OpenAlex
def reconstruct_abstract(inverted) -> str:
    """OpenAlex ne fournit pas le résumé en clair : mot -> positions. On remet les mots dans l'ordre."""
    if not isinstance(inverted, dict):
        return ""
    pairs = []
    for word, positions in inverted.items():
        if isinstance(positions, list):
            pairs.extend((pos, word) for pos in positions if isinstance(pos, int))
    return " ".join(word for _, word in sorted(pairs))


def parse_openalex(payload) -> ParseResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SearchError("openalex : réponse inattendue (pas de liste « results »)")
    out = ParseResult([])
    for work in payload["results"]:
        work = work if isinstance(work, dict) else {}
        openalex_id = str(work.get("id") or "").rsplit("/", 1)[-1]
        primary = work.get("primary_location") or {}
        best = work.get("best_oa_location") or {}
        access = work.get("open_access") or {}
        authors = [
            _text((a.get("author") or {}).get("display_name"))
            for a in work.get("authorships") or []
            if isinstance(a, dict)
        ]
        doi = str(work.get("doi") or "")
        paper = _build(
            id=openalex_id,
            title=_text(work.get("title") or work.get("display_name")),
            abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
            doi=doi,
            source="openalex",
            url=_text(primary.get("landing_page_url")) or doi or str(work.get("id") or ""),
            open_access=bool(access.get("is_oa")),
            year=_year(work.get("publication_year")),
            authors=[a for a in authors if a],
            venue=_text((primary.get("source") or {}).get("display_name")),
            pdf_url=_text(best.get("pdf_url") or primary.get("pdf_url")),
        )
        if paper is None or not openalex_id:
            out.skipped += 1
        else:
            out.papers.append(paper)
    return out


# ---------------------------------------------------------------- Crossref
def clean_jats(raw) -> str:
    """Résumé Crossref : XML JATS -> texte brut."""
    if not raw:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(raw)))
    text = " ".join(text.split())
    return re.sub(r"^abstract\s*[:.]?\s+", "", text, flags=re.IGNORECASE)


def _crossref_year(item: dict) -> int | None:
    for key in ("issued", "published", "published-online", "published-print"):
        parts = (item.get(key) or {}).get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = _year(parts[0][0])
            if year:
                return year
    return None


def parse_crossref(payload) -> ParseResult:
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("items"), list):
        raise SearchError("crossref : réponse inattendue (pas de message.items)")
    out = ParseResult([])
    for item in message["items"]:
        item = item if isinstance(item, dict) else {}
        doi = str(item.get("DOI") or "").strip()
        item_type = item.get("type")
        if item_type and item_type not in CROSSREF_KEEP_TYPES:
            out.skipped += 1
            continue
        titles = item.get("title") or [""]
        containers = item.get("container-title") or [""]
        authors = []
        for a in item.get("author") or []:
            if isinstance(a, dict):
                name = _text(f"{a.get('given', '')} {a.get('family', '')}") or _text(a.get("name"))
                if name:
                    authors.append(name)
        paper = _build(
            id=f"doi-{slug(doi.lower())}",
            title=_text(titles[0] if titles else ""),
            abstract=clean_jats(item.get("abstract")),
            doi=doi,
            source="crossref",
            url=_text(item.get("URL")),
            open_access=False,  # Crossref ne dit rien d'ouvert de façon fiable : on ne l'affirme pas
            year=_crossref_year(item),
            authors=authors,
            venue=_text(containers[0] if containers else ""),
        )
        if paper is None or not doi:
            out.skipped += 1
        else:
            out.papers.append(paper)
    return out


# ---------------------------------------------------------------- Semantic Scholar
def parse_semanticscholar(payload) -> ParseResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("data", []), list):
        raise SearchError("semanticscholar : réponse inattendue (pas de liste « data »)")
    out = ParseResult([])
    for item in payload.get("data") or []:
        item = item if isinstance(item, dict) else {}
        paper_id = str(item.get("paperId") or "").strip()
        external = item.get("externalIds") or {}
        pdf = item.get("openAccessPdf") or {}
        paper = _build(
            id=f"s2-{paper_id}",
            title=_text(item.get("title")),
            abstract=_text(item.get("abstract")),
            doi=str(external.get("DOI") or ""),
            source="semanticscholar",
            url=_text(item.get("url")),
            open_access=bool(item.get("isOpenAccess")),
            year=_year(item.get("year")),
            authors=[_text(a.get("name")) for a in item.get("authors") or [] if isinstance(a, dict) and a.get("name")],
            venue=_text(item.get("venue")),
            pdf_url=_text(pdf.get("url")),
        )
        if paper is None or not paper_id:
            out.skipped += 1
        else:
            out.papers.append(paper)
    return out


# ---------------------------------------------------------------- arXiv (Atom XML)
def parse_arxiv(xml_text: str) -> ParseResult:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raise SearchError("arxiv : réponse XML illisible") from None
    out = ParseResult([])
    for entry in root.findall("a:entry", _ATOM):
        entry_id = _text(entry.findtext("a:id", "", _ATOM))
        if "/api/errors" in entry_id:  # arXiv signale une requête invalide par une « notice » d'erreur
            raise SearchError("arxiv : requête refusée (" + _text(entry.findtext("a:summary", "", _ATOM))[:120] + ")")
        arxiv_id = re.sub(r"v\d+$", "", entry_id.split("/abs/", 1)[-1]) if "/abs/" in entry_id else ""
        published = _text(entry.findtext("a:published", "", _ATOM))
        paper = _build(
            id=f"arxiv-{slug(arxiv_id)}",
            title=_text(entry.findtext("a:title", "", _ATOM)),
            abstract=_text(entry.findtext("a:summary", "", _ATOM)),
            doi=_text(entry.findtext("x:doi", "", _ATOM)),
            source="arxiv",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            open_access=True,
            year=_year(published[:4]),
            authors=[_text(n.text) for n in entry.findall("a:author/a:name", _ATOM) if _text(n.text)],
            venue="arXiv",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        )
        if paper is None or not arxiv_id:
            out.skipped += 1
        else:
            out.papers.append(paper)
    return out


# ---------------------------------------------------------------- exécution d'une requête du plan
def run_query(http: HttpClient, config: SearchConfig, sq: SourceQuery, limit: int) -> ParseResult:
    user_agent = "SLRResearchAI/0.1" + (f" (mailto:{config.contact_email})" if config.contact_email else "")
    headers = {"User-Agent": user_agent}

    if sq.source == "openalex":
        if not config.openalex_api_key:
            raise MissingCredential("openalex : OPENALEX_API_KEY manquante dans .env (clé gratuite obligatoire)")
        params = {sq.query_param: sq.query, **sq.params, "per_page": min(limit, 100),
                  "select": OPENALEX_SELECT, "api_key": config.openalex_api_key}
        if config.contact_email:
            params["mailto"] = config.contact_email
        return parse_openalex(http.get("openalex", OPENALEX_URL, params, headers).json())

    if sq.source == "crossref":
        params = {sq.query_param: sq.query, **sq.params, "rows": min(limit, 100), "select": CROSSREF_SELECT}
        if config.contact_email:
            params["mailto"] = config.contact_email  # « polite pool » : débit plus élevé
        return parse_crossref(http.get("crossref", CROSSREF_URL, params, headers).json())

    if sq.source == "semanticscholar":
        params = {sq.query_param: sq.query, **sq.params, "limit": min(limit, 100), "fields": SEMANTIC_FIELDS}
        if config.semantic_scholar_api_key:
            headers["x-api-key"] = config.semantic_scholar_api_key
        return parse_semanticscholar(http.get("semanticscholar", SEMANTIC_URL, params, headers).json())

    if sq.source == "arxiv":
        params = {sq.query_param: sq.query, "start": 0, "max_results": min(limit, 100), "sortBy": "relevance"}
        return parse_arxiv(http.get("arxiv", ARXIV_URL, params, headers).text)

    raise SearchError(f"source inconnue : {sq.source}")