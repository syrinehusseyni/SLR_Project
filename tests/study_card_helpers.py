"""Aides pour tester 5b : des papiers dont le texte répond à chaque champ, un faux LLM qui cite un vrai chunk,
et un run déjà passé par l'étape 5a (extraction)."""
import json
import re

from slrresearch.core.config import Settings
from slrresearch.core.contract import build_context
from slrresearch.extraction.stage import ExtractionStage
from tests.extraction_helpers import make_pdf, setup_documents


def rich_text(tag: str) -> str:
    """Une page dont les phrases touchent les cinq champs (design, données, modèles, résultats, limites)."""
    return " ".join([
        f"In this paper {tag} we conducted a comparative experiment to evaluate prompting strategies for screening.",
        "The dataset contains 1500 abstracts from 12 systematic reviews in the medical domain.",
        "We tested GPT-4 and LLaMA models with zero-shot prompting and a fine-tuned variant.",
        "Recall reached 0.97 and precision 0.81 on the test set, and the best model outperformed the baseline.",
        "However, a limitation of this work is the risk of hallucination and bias in the outputs.",
    ] * 2)


class RuleLLM:
    """Faux LLM à règle : lit le prompt, cite un vrai passage du premier extrait. `invoke` renvoie du texte (comme FakeLLM)."""

    def __init__(self, rule=None):
        self.rule = rule or self.quote_first_chunk
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.rule(prompt)

    @staticmethod
    def first_chunk(prompt: str) -> tuple[str, str]:
        match = re.search(r"\[CHUNK (?P<id>\S+)\] \[PAGE \d+\]\n(?P<body>.+?)(?:\n\n\[CHUNK |\n\nRÈGLES)", prompt, re.S)
        return match["id"].rstrip("]"), match["body"]

    @classmethod
    def quote_first_chunk(cls, prompt: str) -> str:
        chunk_id, body = cls.first_chunk(prompt)
        quote = " ".join(body.split())[:90]
        return json.dumps({"status": "found", "value": "réponse courte", "chunk_id": chunk_id, "quote": quote})


def make_ctx(store, llm, budget=None, env=None):
    return build_context(store, Settings.from_env(env or {}), budget=budget, make_model=lambda settings, role: llm)


def prepare_run(store, tags=("A", "B"), pages=2, **stage_options):
    """Run avec `documents` + `extraction` prêts (étape 5a réellement exécutée sur de petits PDF)."""
    ctx = setup_documents(store, {tag: make_pdf([rich_text(tag)] * pages) for tag in tags})
    result = ExtractionStage(chunk_size=500, chunk_overlap=80, **stage_options).run(ctx)
    assert result.status == "done", result.message
    return ctx