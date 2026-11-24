"""Normalize the protocol's RQ-specific facets into eight reusable extraction targets."""
import hashlib
import json

from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound
from slrresearch.core.schemas.facet_plan import (
    CanonicalFacet, FacetPlanOutcome, FacetPlanSource,
)
from slrresearch.core.schemas.protocol_plan import Protocol

ARTIFACT = "facet_plan"
VERSION = "facet-plan-v1"

# The mapping is explicit because the current protocol's generated labels are duplicated,
# and several descriptions are filed under the wrong label (for example, RQ2's "study type"
# description is actually about risks). It preserves every original ID and description.
_CATEGORY_BY_SOURCE_ID = {
    "F1": "design", "F2": "metrics", "F3": "models", "F4": "data", "F5": "parameters", "F6": "results",
    "F7": "risks_limits", "F8": "metrics", "F9": "models", "F10": "data", "F11": "risks_limits",
    "F12": "results", "F13": "methodology", "F14": "methodology", "F15": "models", "F16": "data",
    "F17": "parameters", "F18": "results",
}

_DEFINITIONS = {
    "design": ("Type et dessin de l'étude", "Dessin de l'étude des auteurs : expérience, comparaison, benchmark ou autre approche.",
               "study design experimental comparative benchmark methodology screening study", True, "design"),
    "metrics": ("Métriques", "Mesures utilisées pour évaluer l'efficacité, la précision ou la fiabilité du screening.",
                "screening evaluation metrics accuracy precision recall F1 sensitivity specificity reliability", True, "findings"),
    "models": ("Modèles et outils", "LLM, modèles et outils de screening évalués ou utilisés, avec leur rôle.",
               "large language model LLM screening model system tool GPT ChatGPT fine-tuned zero-shot", True, "models"),
    "data": ("Données", "Jeux de données, échantillons, tailles et domaines utilisés dans l'étude.",
             "dataset corpus sample size records titles abstracts systematic reviews domain", False, "data"),
    "parameters": ("Paramètres", "Paramètres d'entraînement, de prompting ou de configuration du modèle.",
                   "hyperparameters prompt settings temperature training epochs fine-tuning configuration", False, ""),
    "results": ("Résultats", "Résultats rapportés qui répondent à la question de recherche associée.",
                "results findings screening performance outcomes efficiency errors comparison", True, "findings"),
    "risks_limits": ("Risques et limites", "Risques, biais, faux négatifs et limites rapportés par les auteurs.",
                     "risks limitations bias false negatives hallucination reliability threats screening", True, "limitations"),
    "methodology": ("Méthodologie et pratiques", "Méthodes d'intégration, workflow et recommandations de bonnes pratiques.",
                    "methodology workflow integration implementation human in the loop guidelines best practices screening", True, ""),
}


def build_facet_plan(protocol: Protocol) -> FacetPlanOutcome:
    source_ids = {facet.id for facet in protocol.facets}
    if source_ids != set(_CATEGORY_BY_SOURCE_ID):
        missing = sorted(set(_CATEGORY_BY_SOURCE_ID) - source_ids)
        unexpected = sorted(source_ids - set(_CATEGORY_BY_SOURCE_ID))
        raise ValueError(f"facettes inattendues pour ce plan (manquantes={missing}, inconnues={unexpected})")

    source_facets = [
        FacetPlanSource(facet_id=f.id, rq_id=f.rq_id, original_label=f.label,
                        original_description=f.description, category=_CATEGORY_BY_SOURCE_ID[f.id])
        for f in protocol.facets
    ]
    definitions = []
    for category, (label, description, query, blocking, reuse_field) in _DEFINITIONS.items():
        sources = [s for s in source_facets if s.category == category]
        definitions.append(CanonicalFacet(
            id=category, label=label, description=description, query_en=query, blocking=blocking,
            reuse_field=reuse_field, source_facet_ids=[s.facet_id for s in sources],
            rq_ids=list(dict.fromkeys(s.rq_id for s in sources)),
        ))

    source_payload = [f.model_dump(mode="json") for f in protocol.facets]
    protocol_hash = hashlib.sha256(json.dumps(source_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return FacetPlanOutcome(version=VERSION, protocol_sha256=protocol_hash,
        gate_id=f"G1:facet_plan:{protocol_hash[:12]}", source_facets=source_facets, facets=definitions)


class FacetPlanStage:
    """Prepare a reviewable plan; the G1 approval must precede any new LLM calls."""

    name = ARTIFACT

    def run(self, ctx) -> StageResult:
        try:
            protocol = ctx.store.read_artifact("protocol", Protocol)
        except ArtifactNotFound as exc:
            return StageResult(status="failed", message=f"artefact protocol introuvable : {exc}")
        try:
            plan = build_facet_plan(protocol)
        except ValueError as exc:
            return StageResult(status="failed", message=str(exc))
        ctx.store.write_artifact(ARTIFACT, FacetPlanOutcome, plan)
        item = ctx.queue.open_item(
            "G1", "facet_plan", plan.gate_id.split(":", 2)[2],
            "Valider le regroupement des 18 facettes source en 8 cibles d'extraction, les descriptions normalisées et les requêtes anglaises dans artifacts/facet_plan.json. Aucun appel LLM de l'étape 6b ne sera fait avant cette validation.",
        )
        ctx.trace.log("facet_plan", "written", source_facets=len(plan.source_facets), canonical_facets=len(plan.facets), gate=item.id)
        if item.status == "open":
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=[item],
                               message="Le plan des facettes attend l'approbation G1.")
        if item.decision != "approved":
            return StageResult(status="failed", artifact=ARTIFACT,
                               message=f"Plan de facettes refusé en G1 : {item.decision}")
        return StageResult(status="done", artifact=ARTIFACT, message="Plan des facettes approuvé; prêt pour l'extraction 6b.")
