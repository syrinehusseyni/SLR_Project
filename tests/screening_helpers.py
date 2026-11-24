"""Fabriques partagées par les tests de l'étape 3 (aucun réseau, aucun vrai LLM)."""
import json

from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol import ResearchQuestion
from slrresearch.core.schemas.protocol_plan import (
    Criteria, Criterion, FacetSpec, Protocol, SearchConcept, SearchPlan, SourceQuery,
)
from slrresearch.core.schemas.search import QueryRun, SearchOutcome, SearchSummary
from tests.fakes import FakeLLM

SETTINGS = Settings.from_env({"LLM_PROVIDER": "ollama"})

TEMPLATE = "SUJET={{topic}}\nINCL={{inclusion}}\nEXCL={{exclusion}}\nTITRE={{title}}\nRESUME={{abstract}}\nAN={{year_from}} LANG={{languages}}"


def make_protocol(topic="LLM et screening", concepts=None, year_from=2018) -> Protocol:
    concepts = concepts or [
        SearchConcept(id="C1", label="méthode", terms=["large language model", "LLM"]),
        SearchConcept(id="C2", label="tâche", terms=["title screening", "abstract screening"]),
    ]
    return Protocol(
        topic=topic,
        questions=[ResearchQuestion(identifier="RQ1", question="Quelle performance obtient-on sur ce sujet ?")],
        concepts=concepts,
        criteria=Criteria(
            inclusion=[Criterion(id="I1", kind="include", text="Étude empirique sur le sujet de la revue")],
            exclusion=[Criterion(id="E1", kind="exclude", text="Opinion sans évaluation empirique")],
            year_from=year_from,
        ),
        facets=[FacetSpec(id="F1", rq_id="RQ1", label="Métrique", description="Mesure de performance")],
        search_plan=SearchPlan(queries=[SourceQuery(source="arxiv", query_param="search_query", query="all:test query")]),
    )


def make_paper(pid="W1", title="LLM for title screening", abstract="We test a large language model for title screening.",
               year=2023) -> Paper:
    return Paper(id=pid, title=title, abstract=abstract, source="openalex", year=year)


def make_search(papers: list[Paper]) -> SearchOutcome:
    n = len(papers)
    return SearchOutcome(
        runs=[QueryRun(source="arxiv", query_param="search_query", query="all:test query", status="ok", retrieved=n)],
        papers=papers,
        found_in={p.id: ["openalex"] for p in papers},
        summary=SearchSummary(per_source={"openalex": n}, retrieved=n, duplicates_removed=0, unique=n,
                              without_abstract=sum(1 for p in papers if not p.abstract.strip())),
    )


def verdict_json(decision="include", confidence=0.9, criteria=("I1",), evidence="large language model for title screening",
                 reasons=("cadre le sujet",)) -> str:
    return json.dumps({"decision": decision, "confidence": confidence, "criteria": list(criteria),
                       "reasons": list(reasons), "evidence": evidence})


def setup_run(store, papers, protocol=None, approve_g1=True, llm_responses=None, budget=None):
    """Écrit protocol + search, ouvre/ferme G1, renvoie (ctx, fake_llm)."""
    from slrresearch.core.schemas.protocol_plan import Protocol as P
    from slrresearch.core.schemas.search import SearchOutcome as S

    protocol = protocol or make_protocol()
    store.write_artifact("protocol", P, protocol.model_dump())
    store.write_artifact("search", S, make_search(papers).model_dump())
    fake = FakeLLM(list(llm_responses or []))
    ctx = build_context(store, SETTINGS, budget=budget, make_model=lambda s, role: fake)
    item = ctx.queue.open_item("G1", "protocol", "main", "valider")
    if approve_g1:
        ctx.queue.close_item(item.id, "approved", "moi", "relu")
    return ctx, fake