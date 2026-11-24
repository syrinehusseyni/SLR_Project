"""File de revue humaine : les 6 portes G1..G6.

Stockage en JSON Lines, en AJOUT SEULEMENT : chaque changement d'etat d'un point
ajoute une ligne (un "instantane"). L'etat courant d'un point = sa derniere ligne.
Rien n'est jamais reecrit, donc l'historique complet est conserve.

L'identifiant d'un point est STABLE : "<porte>:<type de cible>:<id de la cible>".
Consequence importante : si une etape est relancee, elle ne recree pas le meme point,
et un point deja tranche par l'humain n'est pas redemande.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from slrresearch.core.errors import ReviewError
from slrresearch.hitl.decisions import Decision, DecisionLog

Gate = Literal["G1", "G2", "G3", "G4", "G5", "G6"]


class ReviewItem(BaseModel):
    id: str
    gate: Gate
    target_type: str = Field(min_length=1)  # ex: "paper", "protocol", "evidence"
    target_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)  # pourquoi la machine demande un humain
    status: Literal["open", "closed"] = "open"
    decision: str = ""
    decided_by: str = ""
    decided_at: str = ""
    justification: str = ""


class ReviewQueue:
    def __init__(self, path: str | Path, decisions: DecisionLog | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.decisions = decisions

    # -- stockage ----------------------------------------------------------
    def _append(self, item: ReviewItem) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def _load(self) -> dict[str, ReviewItem]:
        items: dict[str, ReviewItem] = {}
        if not self.path.exists():
            return items
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = ReviewItem.model_validate_json(line)
                    items[item.id] = item  # la derniere ligne d'un id l'emporte
        return items

    # -- ouvrir ------------------------------------------------------------
    def open_item(self, gate: str, target_type: str, target_id: str, reason: str) -> ReviewItem:
        """Ouvre un point. Si le meme point existe deja (ouvert OU ferme), le renvoie tel quel."""
        item_id = f"{gate}:{target_type}:{target_id}"
        existing = self._load().get(item_id)
        if existing is not None:
            return existing
        try:
            item = ReviewItem(id=item_id, gate=gate, target_type=target_type, target_id=target_id, reason=reason)
        except ValidationError as exc:
            raise ReviewError(f"point de revue invalide : {exc}") from exc
        self._append(item)
        return item

    # -- fermer ------------------------------------------------------------
    def close_item(self, item_id: str, decision: str, decided_by: str, justification: str) -> ReviewItem:
        """Ferme un point. Decision, auteur ET justification sont obligatoires."""
        item = self._load().get(item_id)
        if item is None:
            raise ReviewError(f"point de revue inconnu : {item_id}")
        if item.status == "closed":
            raise ReviewError(f"point deja ferme : {item_id}")
        for label, value in (("decision", decision), ("decided_by", decided_by), ("justification", justification)):
            if not value or not value.strip():
                raise ReviewError(f"'{label}' obligatoire pour fermer un point de revue")

        closed = item.model_copy(
            update={
                "status": "closed",
                "decision": decision.strip(),
                "decided_by": decided_by.strip(),
                "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "justification": justification.strip(),
            }
        )
        # Journal d'abord : une decision humaine ne doit jamais etre perdue.
        if self.decisions:
            self.decisions.record(
                Decision(
                    item_id=closed.id,
                    gate=closed.gate,
                    target_type=closed.target_type,
                    target_id=closed.target_id,
                    decision=closed.decision,
                    decided_by=closed.decided_by,
                    decided_at=closed.decided_at,
                    justification=closed.justification,
                )
            )
        self._append(closed)
        return closed

    # -- lire --------------------------------------------------------------
    def get(self, item_id: str) -> ReviewItem | None:
        return self._load().get(item_id)

    def list_items(self, status: str | None = None, gate: str | None = None) -> list[ReviewItem]:
        items = list(self._load().values())
        if status is not None:
            items = [i for i in items if i.status == status]
        if gate is not None:
            items = [i for i in items if i.gate == gate]
        return items

    def open_items(self, gate: str | None = None) -> list[ReviewItem]:
        return self.list_items(status="open", gate=gate)

    def has_open(self, gate: str | None = None) -> bool:
        """Vrai si une porte est ouverte : l'orchestrateur doit alors s'arreter."""
        return bool(self.open_items(gate))