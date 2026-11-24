"""Résolveurs LÉGAUX : à partir d'un papier, trouver des adresses de PDF à essayer, dans cet ordre :

1. lien libre déjà connu (OpenAlex / Semantic Scholar / arXiv, selon la source du papier) ;
2. Unpaywall (via le DOI ; adresse e-mail de contact obligatoire dans l'URL) ;
3. Europe PMC (via le DOI, sans clé : liens plein texte en accès libre, articles PMC et preprints) ;
4. Semantic Scholar `openAccessPdf` ;
5. arXiv (identifiant déduit du DOI 10.48550/arXiv.* ou de l'URL) ;
6. liens PDF déclarés par l'éditeur dans Crossref ;
7. page de l'article (https://doi.org/<DOI>) : balise `citation_pdf_url`, que les éditeurs publient pour les moteurs de recherche.

Aucun contournement de paywall ni de protection anti-robot, aucun site pirate : on ne suit que des liens que ces services
publient. Si un site répond par une page de vérification au lieu du PDF, la validation la refuse et le papier va en revue humaine (G3).
Ces fonctions ne téléchargent aucun PDF : elles cherchent des adresses. Une recherche qui échoue
(429, 404...) n'arrête pas le papier : elle est tracée et on passe au résolveur suivant.
Formats vérifiés : Unpaywall (page officielle, nov. 2026) et le test réel du 18 nov. 2026 (OpenAlex, Unpaywall, arXiv, Semantic
Scholar, Crossref). Europe PMC (documentation tierce concordante) et `citation_pdf_url` : à confirmer par un essai réel.
"""
import html
import re
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin

import httpx

from slrresearch.core.schemas.documents import DownloadAttempt
from slrresearch.core.schemas.papers import Paper
from slrresearch.search.http import HttpClient, MIN_INTERVALS, SearchError
from slrresearch.search.sources import SearchConfig

UNPAYWALL_URL = "https://api.unpaywall.org/v2/"
SEMANTIC_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/"
CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_RENDER_URL = "https://europepmc.org/backend/ptpmcrender.fcgi"
DOI_URL = "https://doi.org/"
MAX_LANDING_CHARS = 500_000  # on ne lit que le début de la page de l'article

# Espacement minimal (secondes) : celui de la recherche + Unpaywall + le téléchargement des PDF eux-mêmes.
# "arxiv" couvre aussi arxiv.org/pdf (3 s). "pdf" = tous les autres éditeurs/dépôts (réglable par l'utilisateur).
DOWNLOAD_INTERVALS = {**MIN_INTERVALS, "unpaywall": 0.2, "europepmc": 0.2, "pdf": 1.0}

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:")


@dataclass
class Resolution:
    """Résultat d'un résolveur : des adresses à essayer, et éventuellement la trace d'une recherche sans résultat."""

    resolver: str
    urls: list[str] = field(default_factory=list)
    note: DownloadAttempt | None = None


def clean_doi(doi: str) -> str:
    doi = (doi or "").strip()
    for prefix in _DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip()


def user_agent(config: SearchConfig) -> str:
    return "SLRResearchAI/0.1" + (f" (mailto:{config.contact_email})" if config.contact_email else "")


def _dedupe(urls) -> list[str]:
    seen, out = set(), []
    for url in urls:
        if isinstance(url, str) and url.strip() and url.strip() not in seen:
            seen.add(url.strip())
            out.append(url.strip())
    return out


def _get(http: HttpClient, source: str, resolver: str, url: str, params=None, headers=None):
    """GET d'une page d'API. Renvoie (réponse, None) ou (None, trace de l'échec). BudgetExceeded remonte."""
    try:
        return http.get(source, url, params, headers), None
    except SearchError as exc:
        status = exc.status
        reason = "lookup:no_record" if status == 404 else f"lookup:HTTP {status}" if status else "lookup:network_error"
        return None, DownloadAttempt(resolver=resolver, url=url, http_status=status, validation=reason)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:  # ex. trop de redirections ; jamais str(exc) (l'URL peut contenir un secret)
        return None, DownloadAttempt(resolver=resolver, url=url, validation=f"lookup:{type(exc).__name__}")


def _lookup(http: HttpClient, source: str, resolver: str, url: str, params=None, headers=None):
    """GET d'une API qui répond en JSON. Renvoie (json, None) ou (None, trace de l'échec)."""
    response, failure = _get(http, source, resolver, url, params, headers)
    if failure:
        return None, failure
    try:
        return response.json(), None
    except ValueError:  # JSON invalide
        return None, DownloadAttempt(resolver=resolver, url=url, validation="lookup:invalid_json")


def _no_link(resolver: str, url: str) -> DownloadAttempt:
    return DownloadAttempt(resolver=resolver, url=url, validation="lookup:no_pdf_link")


# ---------------------------------------------------------------- 1. lien déjà connu
_LINK_LABELS = {"openalex": "openalex_oa", "semanticscholar": "s2_open_access", "arxiv": "arxiv", "crossref": "crossref"}


def resolve_known_link(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    label = _LINK_LABELS.get(paper.source, paper.source)
    return Resolution(label, _dedupe([paper.pdf_url]))


# ---------------------------------------------------------------- 2. Unpaywall
def parse_unpaywall(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    locations = [payload.get("best_oa_location")] + list(payload.get("oa_locations") or [])
    return _dedupe(loc.get("url_for_pdf") for loc in locations if isinstance(loc, dict))


def resolve_unpaywall(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    doi = clean_doi(paper.doi)
    if not doi:
        return Resolution("unpaywall")
    if not config.contact_email:
        return Resolution("unpaywall", note=DownloadAttempt(resolver="unpaywall", url="", validation="skipped:no_contact_email"))
    url = UNPAYWALL_URL + quote(doi, safe="/")
    payload, failure = _lookup(http, "unpaywall", "unpaywall", url, {"email": config.contact_email},
                               {"User-Agent": user_agent(config)})
    if failure:
        return Resolution("unpaywall", note=failure)
    urls = parse_unpaywall(payload)
    return Resolution("unpaywall", urls, None if urls else _no_link("unpaywall", url))


# ---------------------------------------------------------------- 3. Europe PMC
_PMCID = re.compile(r"^PMC\d+$")


def parse_europepmc(payload, doi: str) -> list[str]:
    """Liens PDF en accès libre du 1er résultat, à condition que son DOI soit bien celui demandé. Puis l'adresse de rendu PMC."""
    results = (payload.get("resultList") or {}).get("result") if isinstance(payload, dict) else None
    first = results[0] if isinstance(results, list) and results and isinstance(results[0], dict) else None
    if first is None or str(first.get("doi") or "").strip().lower() != doi.lower():
        return []
    entries = (first.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    urls = [e.get("url") for e in entries if isinstance(e, dict)
            and str(e.get("documentStyle") or "").lower() == "pdf" and e.get("availabilityCode") in ("OA", "F")]
    pmcid = str(first.get("pmcid") or "").strip()
    if _PMCID.match(pmcid):
        urls.append(f"{EUROPEPMC_RENDER_URL}?accid={pmcid}&blobtype=pdf")
    return _dedupe(urls)


def resolve_europepmc(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    doi = clean_doi(paper.doi)
    if not doi:
        return Resolution("europepmc")
    params = {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 1}
    payload, failure = _lookup(http, "europepmc", "europepmc", EUROPEPMC_URL, params, {"User-Agent": user_agent(config)})
    if failure:
        return Resolution("europepmc", note=failure)
    urls = parse_europepmc(payload, doi)
    return Resolution("europepmc", urls, None if urls else _no_link("europepmc", EUROPEPMC_URL))


# ---------------------------------------------------------------- 4. Semantic Scholar
def semantic_identifier(paper: Paper) -> str:
    doi = clean_doi(paper.doi)
    if doi:
        return "DOI:" + doi
    if paper.id.startswith("s2-") and len(paper.id) > 3:
        return paper.id[3:]
    return ""


def parse_semantic_open_access(payload) -> list[str]:
    pdf = payload.get("openAccessPdf") if isinstance(payload, dict) else None
    return _dedupe([pdf.get("url")]) if isinstance(pdf, dict) else []


def resolve_semantic_scholar(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    identifier = semantic_identifier(paper)
    if not identifier:
        return Resolution("s2_open_access")
    url = SEMANTIC_PAPER_URL + quote(identifier, safe="/:")
    headers = {"User-Agent": user_agent(config)}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key
    payload, failure = _lookup(http, "semanticscholar", "s2_open_access", url, {"fields": "openAccessPdf"}, headers)
    if failure:
        return Resolution("s2_open_access", note=failure)
    urls = parse_semantic_open_access(payload)
    return Resolution("s2_open_access", urls, None if urls else _no_link("s2_open_access", url))


# ---------------------------------------------------------------- 5. arXiv
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?$", re.IGNORECASE)


def arxiv_id(paper: Paper) -> str:
    """Identifiant arXiv (sans numéro de version) déduit du DOI ou de l'URL du papier, sinon ''."""
    found = _ARXIV_DOI.match(clean_doi(paper.doi))
    raw = found.group(1) if found else ""
    if not raw:
        found = _ARXIV_URL.search(paper.url or "")
        raw = found.group(1) if found else ""
    return re.sub(r"v\d+$", "", raw.strip())


def resolve_arxiv(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    identifier = arxiv_id(paper)
    return Resolution("arxiv", [f"https://arxiv.org/pdf/{identifier}"] if identifier else [])


# ---------------------------------------------------------------- 6. Crossref (liens déclarés par l'éditeur)
def parse_crossref_links(payload) -> list[str]:
    message = payload.get("message") if isinstance(payload, dict) else None
    links = message.get("link") if isinstance(message, dict) else None
    urls = []
    for link in links or []:
        if not isinstance(link, dict):
            continue
        url = str(link.get("URL") or "")
        if "pdf" in str(link.get("content-type") or "").lower() or url.lower().split("?")[0].endswith(".pdf"):
            urls.append(url)
    return _dedupe(urls)


def resolve_crossref(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    doi = clean_doi(paper.doi)
    if not doi:
        return Resolution("crossref")
    url = CROSSREF_WORKS_URL + quote(doi, safe="/")
    params = {"mailto": config.contact_email} if config.contact_email else None
    payload, failure = _lookup(http, "crossref", "crossref", url, params, {"User-Agent": user_agent(config)})
    if failure:
        return Resolution("crossref", note=failure)
    urls = parse_crossref_links(payload)
    return Resolution("crossref", urls, None if urls else _no_link("crossref", url))


# ---------------------------------------------------------------- 7. page de l'article (balise citation_pdf_url)
_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"([A-Za-z_:][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")


def parse_citation_pdf_urls(page: str, base_url: str) -> list[str]:
    """Adresses des balises <meta name="citation_pdf_url" content="..."> (adresses relatives complétées avec `base_url`)."""
    urls = []
    for tag in _META_TAG.findall(page[:MAX_LANDING_CHARS]):
        attributes = {m.group(1).lower(): html.unescape(m.group(2) if m.group(2) is not None else m.group(3))
                      for m in _ATTRIBUTE.finditer(tag)}
        if attributes.get("name", "").strip().lower() == "citation_pdf_url" and attributes.get("content", "").strip():
            full = urljoin(base_url, attributes["content"].strip())
            if full.lower().startswith(("http://", "https://")):
                urls.append(full)
    return _dedupe(urls)


def resolve_landing_page(paper: Paper, http: HttpClient, config: SearchConfig) -> Resolution:
    doi = clean_doi(paper.doi)
    if not doi:
        return Resolution("landing_page")
    url = DOI_URL + quote(doi, safe="/")
    response, failure = _get(http, "pdf", "landing_page", url, None,
                             {"User-Agent": user_agent(config), "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"})
    if failure:
        return Resolution("landing_page", note=failure)
    if response.content[:5] == b"%PDF-":  # le DOI mène directement à un PDF
        return Resolution("landing_page", [str(response.url)])
    urls = parse_citation_pdf_urls(response.text, str(response.url))
    return Resolution("landing_page", urls, None if urls else _no_link("landing_page", url))


# Ordre d'essai (spécification §7.1).
RESOLVERS = (resolve_known_link, resolve_unpaywall, resolve_europepmc, resolve_semantic_scholar, resolve_arxiv,
             resolve_crossref, resolve_landing_page)