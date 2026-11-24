"""Test LIVE (vrai LLM, coût réel) : 20 papiers du run `demo`, mélange avec/sans résumé.

  python -m pytest tests/live/test_screening_live.py -m live -s
"""
import pytest

from slrresearch.core.budgets import Budget
from slrresearch.core.config import load_settings
from slrresearch.core.contract import build_context
from slrresearch.core.schemas.protocol_plan import Protocol
from slrresearch.core.schemas.search import SearchOutcome
from slrresearch.core.store import RunStore
from slrresearch.screening.arbitration import decide
from slrresearch.screening.llm_screener import load_prompt, screen_paper
from slrresearch.screening.rules import apply_rules

pytestmark = pytest.mark.live


def test_live_screening_sample():
    store = RunStore("runs", "demo")
    if not (store.exists("protocol") and store.exists("search")):
        pytest.skip("run demo absent")
    protocol = store.read_artifact("protocol", Protocol)
    papers = store.read_artifact("search", SearchOutcome).papers
    with_abstract = [p for p in papers if p.abstract.strip()][:17]
    sample = with_abstract + [p for p in papers if not p.abstract.strip()][:3]
    ctx = build_context(store, load_settings(), budget=Budget(max_calls=60))
    llm, template = ctx.llm("screening"), load_prompt()
    statuses = []
    for paper in sample:
        rule = apply_rules(paper, protocol)
        if rule.decisive or not paper.abstract.strip():
            print(f"{paper.id:>12} | A={rule.decision:9} | LLM non appelé")
            continue
        result = screen_paper(llm, paper, protocol, template, ctx.trace)
        final = decide(rule, result.status, result.verdict)
        statuses.append(result.status)
        v = result.verdict
        print(f"{paper.id:>12} | A={rule.decision:9} | B={(v.decision if v else '-'):9} | {result.status:16} | {final[1]}")
    assert statuses, "aucun appel LLM effectué"
    ok = statuses.count("ok")
    print(f"statuts : {ok}/{len(statuses)} ok")
    assert ok >= len(statuses) // 2, "plus de la moitié des réponses LLM sont inutilisables : revoir le prompt ou le modèle"