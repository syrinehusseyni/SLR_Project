"""Stockage des artefacts d'un run : runs/<run_id>/artifacts/<nom>.json

Trois garanties :
  1. ECRITURE VALIDEE : on refuse d'ecrire un artefact qui ne respecte pas son schema.
  2. ECRITURE ATOMIQUE : fichier temporaire puis os.replace -> jamais de JSON a moitie ecrit
     si le programme s'arrete en plein milieu.
  3. SAUVEGARDE .bak : l'ancienne version est gardee avant d'etre ecrasee.
"""
import json
import os
import re
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from slrresearch.core.errors import ArtifactInvalid, ArtifactNotFound

T = TypeVar("T", bound=BaseModel)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")  # empeche "../../x" (path traversal)


class RunStore:
    def __init__(self, root: str | Path, run_id: str):
        if not _SAFE_NAME.match(run_id):
            raise ValueError(f"run_id invalide : {run_id!r}")
        self.run_id = run_id
        self.run_dir = Path(root) / run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -- chemins -----------------------------------------------------------
    @property
    def trace_path(self) -> Path:
        return self.run_dir / "trace.jsonl"

    @property
    def hitl_path(self) -> Path:
        return self.run_dir / "hitl.jsonl"

    def artifact_path(self, name: str) -> Path:
        if not _SAFE_NAME.match(name):
            raise ValueError(f"nom d'artefact invalide : {name!r}")
        return self.artifacts_dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.artifact_path(name).exists()

    # -- ecriture ----------------------------------------------------------
    def write_artifact(self, name: str, schema: type[T], data: BaseModel | dict) -> Path:
        """Valide `data` avec `schema`, puis ecrit de facon atomique (avec .bak)."""
        raw = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        try:
            validated = schema.model_validate(raw)
        except ValidationError as exc:
            raise ArtifactInvalid(f"artefact '{name}' invalide : {exc}") from exc

        path = self.artifact_path(name)
        if path.exists():
            shutil.copy2(path, path.with_suffix(".json.bak"))

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)  # atomique, fonctionne aussi sous Windows
        return path

    # -- lecture -----------------------------------------------------------
    def read_artifact(self, name: str, schema: type[T]) -> T:
        """Lit et VALIDE : un fichier modifie a la main et casse est detecte ici."""
        path = self.artifact_path(name)
        if not path.exists():
            raise ArtifactNotFound(f"artefact '{name}' absent du run '{self.run_id}'")
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ArtifactInvalid(f"artefact '{name}' corrompu : {exc}") from exc