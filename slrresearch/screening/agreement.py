"""Accord entre l'agent A et l'agent B : accord brut et kappa de Cohen (3 catégories)."""
from slrresearch.core.schemas.screening import AgreementStats

CATEGORIES = ("include", "exclude", "uncertain")


def cohen_kappa(pairs: list[tuple[str, str]]) -> AgreementStats:
    n = len(pairs)
    matrix: dict[str, int] = {}
    for a, b in pairs:
        key = f"{a}|{b}"
        matrix[key] = matrix.get(key, 0) + 1
    if n == 0:
        return AgreementStats(n=0, agreed=0, note="aucune paire A/B comparable")
    agreed = sum(1 for a, b in pairs if a == b)
    observed = agreed / n
    expected = sum(
        (sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n) for c in CATEGORIES
    )
    if expected >= 1.0:
        return AgreementStats(n=n, agreed=agreed, observed=observed, matrix=matrix,
                              note="kappa indéfini : les deux agents ont toujours donné la même catégorie unique")
    return AgreementStats(n=n, agreed=agreed, observed=observed, kappa=(observed - expected) / (1 - expected), matrix=matrix)