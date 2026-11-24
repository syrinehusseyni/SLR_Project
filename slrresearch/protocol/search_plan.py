"""Requêtes par source, construites par le PROGRAMME (jamais par le LLM).

Syntaxes vérifiées dans les documentations officielles le 13 nov. 2026 (non testées en réel : étape 2) :
- OpenAlex : paramètre `search.title_and_abstract` ; AND/OR/NOT en MAJUSCULES, guillemets doubles
  pour les phrases, parenthèses pour l'ordre. Filtre de date : `filter=from_publication_date:AAAA-01-01`.
- arXiv : `search_query` ; champs préfixés (`all:`), AND / OR / ANDNOT, parenthèses, guillemets doubles.
  Le client devra encoder (espace -> +, ( -> %28, " -> %22). Filtre : `submittedDate:[AAAAMMJJHHMM TO ...]`.
- Crossref : AUCUN opérateur booléen (le score de pertinence trie) ; texte libre dans `query` ;
  filtre `from-pub-date:AAAA-01-01`.
- Semantic Scholar : /paper/search = mots-clés simples, pas de syntaxe booléenne (elle n'existe que pour
  /paper/search/bulk) ; filtre `year=AAAA-`.
"""
import re

from slrresearch.core.schemas.protocol_plan import SearchConcept, SearchPlan, SourceQuery

_SIMPLE_WORD = re.compile(r"[A-Za-z0-9]+")


def quote_term(term: str) -> str:
    """Mot simple tel quel ; sinon (espaces, tirets...) entre guillemets doubles."""
    return term if _SIMPLE_WORD.fullmatch(term) else f'"{term}"'


def boolean_query(concepts: list[SearchConcept]) -> str:
    """(t1 OR t2) AND (t3 OR t4) — syntaxe OpenAlex."""
    groups = ["(" + " OR ".join(quote_term(t) for t in c.terms) + ")" for c in concepts]
    return " AND ".join(groups)


def arxiv_query(concepts: list[SearchConcept], year_from: int | None = None) -> str:
    groups = ["(" + " OR ".join(f"all:{quote_term(t)}" for t in c.terms) + ")" for c in concepts]
    query = " AND ".join(groups)
    if year_from is not None:
        query += f" AND submittedDate:[{year_from}01010000 TO 209912312359]"
    return query


def keyword_variants(concepts: list[SearchConcept], max_variants: int = 3) -> list[str]:
    """Requêtes en mots simples (sources sans booléen).

    Variante i = i-ième terme de chaque concept (le premier si le concept n'en a pas autant).
    """
    variants: list[str] = []
    for i in range(max_variants):
        if i > 0 and not any(len(c.terms) > i for c in concepts):
            break
        words = [c.terms[i] if i < len(c.terms) else c.terms[0] for c in concepts]
        query = " ".join(words)
        if query not in variants:
            variants.append(query)
    return variants


def build_search_plan(concepts: list[SearchConcept], year_from: int | None = None, max_variants: int = 3) -> SearchPlan:
    queries: list[SourceQuery] = []

    openalex_params = {"filter": f"from_publication_date:{year_from}-01-01"} if year_from else {}
    queries.append(
        SourceQuery(
            source="openalex",
            query_param="search.title_and_abstract",
            query=boolean_query(concepts),
            params=openalex_params,
            note="Booléen complet (AND/OR en majuscules, phrases entre guillemets), titre + résumé.",
        )
    )
    queries.append(
        SourceQuery(
            source="arxiv",
            query_param="search_query",
            query=arxiv_query(concepts, year_from),
            note="Booléen sur tous les champs ; la valeur doit être encodée pour l'URL par le client.",
        )
    )
    for text in keyword_variants(concepts, max_variants):
        queries.append(
            SourceQuery(
                source="crossref",
                query_param="query",
                query=text,
                params={"filter": f"from-pub-date:{year_from}-01-01"} if year_from else {},
                note="Crossref n'a pas d'opérateurs booléens : requête libre triée par pertinence, précision plus faible.",
            )
        )
    for text in keyword_variants(concepts, max_variants):
        queries.append(
            SourceQuery(
                source="semanticscholar",
                query_param="query",
                query=text,
                params={"year": f"{year_from}-"} if year_from else {},
                note="Recherche par pertinence en mots-clés simples (pas de syntaxe booléenne sur /paper/search).",
            )
        )
    return SearchPlan(queries=queries)