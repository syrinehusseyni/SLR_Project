"""Appels RÉELS aux 4 API (nécessite le réseau et, pour OpenAlex, OPENALEX_API_KEY dans .env).

python -m pytest -m live tests/live/test_search_live.py -v -s
Les parseurs sont écrits d'après la documentation : ce test confirme (ou infirme) qu'ils lisent bien les vraies réponses.
"""
import os

import httpx
import pytest
from dotenv import dotenv_values

from slrresearch.core.budgets import Budget
from slrresearch.core.schemas.protocol_plan import SearchConcept
from slrresearch.protocol.search_plan import build_search_plan
from slrresearch.search.http import HttpClient, MissingCredential, SearchError
from slrresearch.search.sources import SearchConfig, run_query


@pytest.mark.live
@pytest.mark.parametrize("source", ["openalex", "crossref", "semanticscholar", "arxiv"])
def test_real_source_returns_parsable_papers(source):
    config = SearchConfig.from_env({**dotenv_values(".env"), **os.environ})
    concepts = [SearchConcept(id="C1", label="method", terms=["large language model", "LLM"]),
                SearchConcept(id="C2", label="task", terms=["title and abstract screening", "citation screening"])]
    sq = next(q for q in build_search_plan(concepts, 2022).queries if q.source == source)
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        http = HttpClient(client, Budget(max_calls=10))
        try:
            result = run_query(http, config, sq, 10)
        except MissingCredential as exc:
            pytest.skip(str(exc))
        except SearchError as exc:
            if exc.status == 429:  # limite de débit : ce n'est pas un défaut du parseur
                pytest.skip(f"{source} limite le débit (429) : réessayez plus tard ou ajoutez une clé API")
            raise
    print(f"\n{source}: {len(result.papers)} papiers, {result.skipped} écartés")
    for paper in result.papers[:3]:
        print("  -", paper.id, "|", paper.title[:70], "|", paper.year, "| résumé:", bool(paper.abstract))
    assert result.papers, f"{source} n'a renvoyé aucune notice exploitable"