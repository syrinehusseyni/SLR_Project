"""Faux LLM partage par tous les tests : aucun reseau, aucun cout, resultat deterministe."""


class FakeLLM:
    """Renvoie des reponses pre-ecrites dans l'ordre et compte ses appels."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM : plus de reponse prevue (appel inattendu)")
        return self.responses.pop(0)

class FlakyLLM:
    """Echoue `failures` fois (comme une API qui renvoie 429), puis repond."""

    def __init__(self, failures: int, answer: str = "ok"):
        self.failures = failures
        self.answer = answer
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("429 trop de requetes")
        return self.answer