"""Agent A : règles déterministes, sans LLM. Tout vient du protocole approuvé, rien du sujet.

Ordre : année (règle objective) -> résumé absent -> couverture des concepts de recherche.
A ne peut exclure « sûrement » que par l'année ; sinon il propose, et le LLM ou l'humain tranche.
"""
import re

from slrresearch.core.schemas.papers import Paper
from slrresearch.core.schemas.protocol_plan import Protocol, SearchConcept
from slrresearch.core.schemas.screening import RuleVerdict
from slrresearch.screening.text import normalize_for_matching


def _term_pattern(term: str):
    """Mot ou expression entière (« ai » ne matche pas « chain »), pluriel simple accepté."""
    normalized = normalize_for_matching(term)
    if not normalized:
        return None
    return re.compile(rf"(?<!\w){re.escape(normalized)}(?:s|es)?(?!\w)")


def matched_terms(text_norm: str, concept: SearchConcept) -> list[str]:
    found = []
    for term in concept.terms:
        pattern = _term_pattern(term)
        if pattern is not None and pattern.search(text_norm):
            found.append(term)
    return found


def apply_rules(paper: Paper, protocol: Protocol) -> RuleVerdict:
    year_from = protocol.criteria.year_from
    if year_from is not None and paper.year is not None and paper.year < year_from:
        return RuleVerdict(
            paper_id=paper.id, decision="exclude", decisive=True,
            reasons=[f"Année {paper.year} antérieure à l'année minimale {year_from}"],
        )
    if not paper.abstract.strip():
        return RuleVerdict(
            paper_id=paper.id, decision="uncertain",
            reasons=["Résumé absent : décision impossible sans revue humaine"],
        )

    text_norm = normalize_for_matching(f"{paper.title} {paper.abstract}")
    hits = {c.id: matched_terms(text_norm, c) for c in protocol.concepts}
    covered = [cid for cid, terms in hits.items() if terms]
    missing = [cid for cid, terms in hits.items() if not terms]

    if not missing:
        detail = "; ".join(f"{cid} ({', '.join(hits[cid])})" for cid in covered)
        return RuleVerdict(paper_id=paper.id, decision="include", concepts_matched=covered,
                           reasons=[f"Tous les concepts de recherche sont présents : {detail}"])
    if not covered:
        return RuleVerdict(paper_id=paper.id, decision="exclude",
                           reasons=["Aucun concept de recherche présent dans le titre ou le résumé"])
    return RuleVerdict(paper_id=paper.id, decision="uncertain", concepts_matched=covered,
                       reasons=[f"Concepts présents : {', '.join(covered)} ; absents : {', '.join(missing)}"])