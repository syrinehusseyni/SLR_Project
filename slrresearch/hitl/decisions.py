"""Journal des decisions humaines (JSON Lines, en ajout seulement).

Qui a decide, quand, quoi et POURQUOI. Ce journal sert ensuite a l'evaluation
(accord humain / machine) et a l'honnetete methodologique du rapport.
"""
from pathlib import Path

from pydantic import BaseModel


class Decision(BaseModel):
    item_id: str
    gate: str
    target_type: str
    target_id: str
    decision: str
    decided_by: str
    decided_at: str
    justification: str


class DecisionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: Decision) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(decision.model_dump_json() + "\n")

    def read(self) -> list[Decision]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [Decision.model_validate_json(line) for line in f if line.strip()]