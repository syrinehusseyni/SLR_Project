"""Aides communes aux tests du téléchargement : faux serveur HTTP et papiers de test."""
import httpx

from slrresearch.core.budgets import Budget
from slrresearch.core.schemas.papers import Paper
from slrresearch.search.http import HttpClient, RateLimiter
from slrresearch.search.sources import SearchConfig


def make_http(handler, max_calls=50, max_retries=0, intervals=None):
    """HttpClient branché sur un faux serveur (aucun réseau). Renvoie (http, liste des requêtes reçues, pauses)."""
    requests, sleeps = [], []

    def recording(request):
        requests.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(recording))
    http = HttpClient(client, Budget(max_calls=max_calls), None, max_retries=max_retries,
                      limiter=RateLimiter(intervals or {}, sleep=sleeps.append), sleep=sleeps.append)
    return http, requests, sleeps


def make_paper(**kw):
    fields = dict(id="W1", title="Automated title and abstract screening for scoping reviews using the GPT-4 Large Language Model",
                  source="openalex")
    fields.update(kw)
    return Paper(**fields)


def config(email="moi@exemple.org", s2_key=""):
    return SearchConfig(contact_email=email, semantic_scholar_api_key=s2_key)


def pdf_response(data):
    return httpx.Response(200, content=data, headers={"content-type": "application/pdf"})