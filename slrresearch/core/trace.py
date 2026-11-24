"""Trace unique (JSON Lines) partagee par tous les agents.

Un evenement = une ligne JSON. On AJOUTE toujours (mode "a"), on ne reecrit jamais :
la trace est l'historique fiable de ce que chaque agent a decide (PLAN -> ... -> DECIDE).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, agent: str, event: str, **data: Any) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": agent,
            "event": event,
            "data": data,
        }
        # default=str : evite de planter si un objet non serialisable se glisse dans data
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]