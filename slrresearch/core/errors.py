"""Erreurs explicites du projet.

Principe : une erreur doit dire CE QUI s'est passe, pas seulement "ca a plante".
Toutes heritent de SLRError, ce qui permet d'attraper "toute erreur du projet"
sans attraper les bugs Python ordinaires (TypeError, etc.).
"""


class SLRError(Exception):
    """Racine de toutes les erreurs du projet."""


class BudgetExceeded(SLRError):
    """Un budget (appels, tokens, secondes) est epuise : l'agent doit s'arreter."""


class ArtifactInvalid(SLRError):
    """Un artefact ne respecte pas son schema Pydantic (a l'ecriture ou a la lecture)."""


class ArtifactNotFound(SLRError):
    """L'artefact demande n'existe pas dans le dossier du run."""

class ConfigError(SLRError):
    """Configuration invalide ou incomplete (.env : provider inconnu, cle manquante...)."""


class LLMError(SLRError):
    """Un appel LLM a echoue meme apres tous les essais."""

class ReviewError(SLRError):
    """Operation interdite sur la file de revue humaine (ex: fermer sans justification)."""