"""Étape C : accord ou humain. Un `uncertain` n'est jamais une décision : il part en porte G2."""
from slrresearch.core.schemas.screening import RuleVerdict, ScreeningVerdict

DEFAULT_MIN_CONFIDENCE = 0.6


def decide(rule: RuleVerdict, llm_status: str, verdict: ScreeningVerdict | None,
           min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> tuple[str, str, str]:
    """Renvoie (décision finale, origine, raison d'escalade). origine `pending` => décision `uncertain`."""
    if rule.decisive:
        return rule.decision, "rule", ""
    if llm_status == "not_run":
        return "uncertain", "pending", rule.reasons[0]
    if llm_status in ("invalid_format", "llm_error"):
        return "uncertain", "pending", f"LLM inutilisable pour ce papier ({llm_status})"
    if llm_status == "grounding_failed":
        return "uncertain", "pending", "Preuve citée par le LLM absente du titre et du résumé"
    assert verdict is not None  # garanti par LLMResult
    if rule.decision != verdict.decision:
        return "uncertain", "pending", f"Désaccord entre la règle ({rule.decision}) et le LLM ({verdict.decision})"
    if verdict.decision == "uncertain":
        return "uncertain", "pending", "Les deux agents s'abstiennent"
    if verdict.confidence < min_confidence:
        return "uncertain", "pending", f"Confiance du LLM {verdict.confidence:.2f} inférieure au seuil {min_confidence:.2f}"
    return verdict.decision, "agreement", ""