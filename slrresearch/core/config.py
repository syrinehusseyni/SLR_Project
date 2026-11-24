"""Configuration : lit le .env A LA DEMANDE (pas a l'import) et choisit le modele par role.

Changer de fournisseur = changer UNE ligne du .env :  LLM_PROVIDER=ollama  ou  groq.
Chaque fournisseur a ses propres modeles, donc rien d'autre a modifier.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from slrresearch.core.errors import ConfigError

PROVIDERS = ("ollama", "groq")
# Un modele par ROLE : le juge de fidelite doit pouvoir etre different de l'extracteur.
ROLES = ("screening", "extraction", "judge", "writer", "planner")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # a verifier sur la liste des modeles Groq
    role_models: dict = field(default_factory=dict)  # surcharges pour le fournisseur actif
    timeout_seconds: float = 120.0
    contact_email: str = ""
    openalex_api_key: str = ""
    semantic_scholar_api_key: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        def get(key: str, default: str = "") -> str:
            return (env.get(key) or default).strip()

        provider = get("LLM_PROVIDER", "ollama").lower()
        if provider not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER={provider!r} inconnu ; valeurs possibles : {PROVIDERS}")

        try:
            timeout = float(get("LLM_TIMEOUT", "120"))
        except ValueError as exc:
            raise ConfigError("LLM_TIMEOUT doit etre un nombre (secondes)") from exc

        # Surcharges par role, lues uniquement pour le fournisseur ACTIF :
        #   OLLAMA_MODEL_JUDGE=...   /   GROQ_MODEL_JUDGE=...
        role_models = {}
        for role in ROLES:
            value = get(f"{provider.upper()}_MODEL_{role.upper()}")
            if value:
                role_models[role] = value

        return cls(
            llm_provider=provider,
            ollama_base_url=get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=get("OLLAMA_MODEL", "llama3.1:8b"),
            groq_api_key=get("GROQ_API_KEY"),
            groq_model=get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            role_models=role_models,
            timeout_seconds=timeout,
            contact_email=get("CONTACT_EMAIL"),
            openalex_api_key=get("OPENALEX_API_KEY"),
            semantic_scholar_api_key=get("SEMANTIC_SCHOLAR_API_KEY"),
        )

    @property
    def default_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.groq_model

    def model_for(self, role: str) -> str:
        if role not in ROLES:
            raise ConfigError(f"role inconnu : {role!r} ; roles possibles : {ROLES}")
        return self.role_models.get(role) or self.default_model

    @property
    def judge_is_independent(self) -> bool:
        """Vrai si le juge de fidelite n'est pas le meme modele que l'extracteur."""
        return self.model_for("judge") != self.model_for("extraction")


def load_settings(dotenv_path: str | Path = ".env", environ: Mapping[str, str] | None = None) -> Settings:
    """Lit le fichier .env, puis les variables d'environnement (qui ont priorite)."""
    from dotenv import dotenv_values

    values: dict[str, str] = {}
    path = Path(dotenv_path)
    if path.exists():
        values.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    values.update(os.environ if environ is None else environ)
    return Settings.from_env(values)


def build_model(settings: Settings, role: str):
    """Construit le modele LangChain (temperature=0) du fournisseur actif pour ce role.

    Les imports sont faits ICI : les tests et les autres modules n'ont pas besoin
    d'avoir langchain installe tant qu'on ne construit pas de vrai modele.
    """
    name = settings.model_for(role)
    try:
        if settings.llm_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=name,
                base_url=settings.ollama_base_url,
                temperature=0,
                client_kwargs={"timeout": settings.timeout_seconds},
            )

        if not settings.groq_api_key:
            raise ConfigError("GROQ_API_KEY manquante dans .env (necessaire avec LLM_PROVIDER=groq)")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=name,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=settings.timeout_seconds,
        )
    except ImportError as exc:
        pkg = "langchain-ollama" if settings.llm_provider == "ollama" else "langchain-groq"
        raise ConfigError(f"paquet manquant : pip install {pkg}") from exc