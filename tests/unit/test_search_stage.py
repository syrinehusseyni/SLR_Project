import httpx

from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.protocol import ResearchQuestion
from slrresearch.core.schemas.protocol_plan import (
    Criteria, Criterion, FacetSpec, Protocol, SearchConcept,
)
from slrresearch.core.schemas.search import SearchOutcome
from slrresearch.protocol.search_plan import build_search_plan
from slrresearch.search.sources import SearchConfig
from slrresearch.search.stage import SearchStage
from tests.fakes import FakeLLM
from tests.unit.test_search_sources import ARXIV, CROSSREF, OPENALEX, SEMANTIC

KEYS = SearchConfig(openalex_api_key="OA-SECRET", semantic_scholar_api_key="S2-SECRET", contact_email="me@example.org")


def make_protocol(store, approve=True):
    concepts = [SearchConcept(id="C1", label="method", terms=["large language model", "LLM"]),
                SearchConcept(id="C2", label="task", terms=["title screening", "citation screening"])]
    protocol = Protocol(
        topic="LLM et screening",
        questions=[ResearchQuestion(identifier="RQ1", question="Quelle précision obtiennent les LLM au screening ?")],
        concepts=concepts,
        criteria=Criteria(inclusion=[Criterion(id="I1", kind="include", text="Évalue un LLM pour le screening")],
                          exclusion=[Criterion(id="E1", kind="exclude", text="Opinion sans évaluation empirique")], year_from=2018),
        facets=[FacetSpec(id="F1", rq_id="RQ1", label="Métrique", description="Mesure de performance")],
        search_plan=build_search_plan(concepts, 2018),
        generation={"concepts": "accepted", "criteria": "accepted", "facets": "accepted"},
    )
    store.write_artifact("protocol", Protocol, protocol.model_dump())
    ctx = build_context(store, Settings.from_env({"LLM_PROVIDER": "ollama"}), make_model=lambda s, r: FakeLLM([]))
    item = ctx.queue.open_item("G1", "protocol", "main", "valider")
    if approve:
        ctx.queue.close_item(item.id, "approved", "moi", "relu")
    return ctx, protocol


class Server:
    """Faux serveur : répond selon l'hôte et enregistre les requêtes reçues."""
    def __init__(self, overrides=None):
        self.requests, self.overrides = [], overrides or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host in self.overrides:
            return self.overrides[host](request)
        if host == "api.openalex.org":
            return httpx.Response(200, json=OPENALEX)
        if host == "api.crossref.org":
            return httpx.Response(200, json=CROSSREF)
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json=SEMANTIC)
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV)
        return httpx.Response(404)

    def hosts(self):
        return [r.url.host for r in self.requests]


def stage(server, **kw):
    return SearchStage(KEYS, transport=httpx.MockTransport(server), sleep=lambda s: None, **kw)


def test_requires_protocol(store):
    ctx = build_context(store, Settings.from_env({}), make_model=lambda s, r: FakeLLM([]))
    r = stage(Server()).run(ctx)
    assert r.status == "failed" and "protocol" in r.message


def test_requires_g1_approved(store):
    ctx, _ = make_protocol(store, approve=False)
    server = Server()
    r = stage(server).run(ctx)
    assert r.status == "failed" and "G1" in r.message and server.requests == []


def test_full_search_writes_validated_artifact(store):
    ctx, protocol = make_protocol(store)
    server = Server()
    r = stage(server).run(ctx)
    assert r.status == "done" and r.artifact == "search"
    outcome = store.read_artifact("search", SearchOutcome)
    assert len(outcome.runs) == len(protocol.search_plan.queries) and all(run.status == "ok" for run in outcome.runs)
    s = outcome.summary
    assert s.retrieved == s.unique + s.duplicates_removed and s.duplicates_removed > 0
    assert set(s.per_source) == {"openalex", "crossref", "semanticscholar", "arxiv"}
    assert s.without_abstract >= 1
    # la même notice trouvée par plusieurs requêtes est dédoublonnée et sa provenance conservée
    w111 = next(p for p in outcome.papers if p.id == "W111")
    assert w111.abstract == "We test LLM test screening" and outcome.found_in["W111"] == ["openalex"]


def test_credentials_and_params_are_sent_correctly(store):
    ctx, protocol = make_protocol(store)
    server = Server()
    stage(server).run(ctx)
    oa = next(r for r in server.requests if r.url.host == "api.openalex.org")
    q = dict(oa.url.params)
    assert q["api_key"] == "OA-SECRET" and q["mailto"] == "me@example.org" and q["per_page"] == "50"
    assert q["filter"] == "from_publication_date:2018-01-01"
    assert q["search.title_and_abstract"].startswith('("large language model" OR LLM) AND')
    s2 = next(r for r in server.requests if r.url.host == "api.semanticscholar.org")
    assert s2.headers["x-api-key"] == "S2-SECRET" and dict(s2.url.params)["year"] == "2018-"
    cr = next(r for r in server.requests if r.url.host == "api.crossref.org")
    assert dict(cr.url.params)["filter"] == "from-pub-date:2018-01-01" and dict(cr.url.params)["mailto"] == "me@example.org"
    ax = next(r for r in server.requests if r.url.host == "export.arxiv.org")
    assert "all:" in dict(ax.url.params)["search_query"]
    assert "SECRET" not in store.trace_path.read_text(encoding="utf-8")  # aucune clé dans la trace


def test_missing_openalex_key_skips_that_source_only(store):
    ctx, _ = make_protocol(store)
    server = Server()
    r = SearchStage(SearchConfig(), transport=httpx.MockTransport(server), sleep=lambda s: None).run(ctx)
    assert r.status == "done"
    outcome = store.read_artifact("search", SearchOutcome)
    oa = [run for run in outcome.runs if run.source == "openalex"]
    assert oa and oa[0].status == "skipped" and "OPENALEX_API_KEY" in oa[0].error
    assert "api.openalex.org" not in server.hosts()


def test_one_source_down_does_not_stop_the_run(store):
    ctx, _ = make_protocol(store)
    server = Server({"api.crossref.org": lambda r: httpx.Response(500)})
    assert stage(server, max_http_calls=200).run(ctx).status == "done"
    outcome = store.read_artifact("search", SearchOutcome)
    statuses = {(run.source, run.status) for run in outcome.runs}
    assert ("crossref", "error") in statuses and ("openalex", "ok") in statuses


def test_rate_limited_source_is_labelled(store):
    ctx, _ = make_protocol(store)
    server = Server({"api.semanticscholar.org": lambda r: httpx.Response(429)})
    assert stage(server, max_http_calls=200).run(ctx).status == "done"
    outcome = store.read_artifact("search", SearchOutcome)
    assert any(run.source == "semanticscholar" and run.status == "rate_limited" for run in outcome.runs)


def test_budget_exhaustion_skips_the_rest_and_keeps_what_was_found(store):
    ctx, protocol = make_protocol(store)
    server = Server()
    assert stage(server, max_http_calls=2).run(ctx).status == "done"
    outcome = store.read_artifact("search", SearchOutcome)
    assert len(server.requests) == 2
    skipped = [run for run in outcome.runs if run.status == "skipped"]
    assert len(skipped) == len(protocol.search_plan.queries) - 2 and "budget" in skipped[0].error


def test_everything_fails_gives_failed_and_no_artifact(store):
    ctx, _ = make_protocol(store)
    server = Server({h: (lambda r: httpx.Response(500)) for h in
                     ("api.openalex.org", "api.crossref.org", "api.semanticscholar.org", "export.arxiv.org")})
    r = stage(server, max_http_calls=500).run(ctx)
    assert r.status == "failed" and "aucun résultat" in r.message
    assert not store.exists("search")


def test_artifact_is_reused_then_force_redoes_it(store):
    ctx, _ = make_protocol(store)
    server = Server()
    stage(server).run(ctx)
    n = len(server.requests)
    assert stage(server).run(ctx).status == "done" and len(server.requests) == n   # aucun nouvel appel
    assert stage(server, force=True).run(ctx).status == "done" and len(server.requests) == 2 * n


def test_trace_has_summary(store):
    ctx, _ = make_protocol(store)
    stage(Server()).run(ctx)
    events = {e["event"] for e in ctx.trace.read()}
    assert {"query_ok", "written", "http_get"} <= events