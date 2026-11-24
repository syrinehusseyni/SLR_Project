import pytest

from pydantic import ValidationError

from slrresearch.core.schemas.screening import RuleVerdict, ScreeningRecord, ScreeningVerdict
from slrresearch.screening.agreement import cohen_kappa
from slrresearch.screening.arbitration import decide


def rule(decision="include", decisive=False):
    return RuleVerdict(paper_id="W1", decision=decision, reasons=["r"], decisive=decisive)


def verdict(decision="include", confidence=0.9):
    return ScreeningVerdict(paper_id="W1", decision=decision, confidence=confidence, reasons=["r"])


def test_decisive_rule_wins_without_llm():
    assert decide(rule("exclude", True), "not_run", None) == ("exclude", "rule", "")


def test_missing_abstract_goes_to_human():
    r = RuleVerdict(paper_id="W1", decision="uncertain", reasons=["Résumé absent"])
    final, origin, why = decide(r, "not_run", None)
    assert (final, origin) == ("uncertain", "pending") and "Résumé absent" in why


def test_agreement_with_good_confidence_is_automatic():
    assert decide(rule("include"), "ok", verdict("include", 0.9)) == ("include", "agreement", "")


def test_disagreement_goes_to_human():
    final, origin, why = decide(rule("include"), "ok", verdict("exclude"))
    assert (final, origin) == ("uncertain", "pending") and "Désaccord" in why


def test_both_uncertain_goes_to_human():
    assert decide(rule("uncertain"), "ok", verdict("uncertain"))[1] == "pending"


def test_low_confidence_goes_to_human_and_threshold_is_configurable():
    assert decide(rule("include"), "ok", verdict("include", 0.5), 0.6)[1] == "pending"
    assert decide(rule("include"), "ok", verdict("include", 0.5), 0.4)[1] == "agreement"


@pytest.mark.parametrize("status", ["invalid_format", "llm_error", "grounding_failed"])
def test_unusable_llm_goes_to_human(status):
    assert decide(rule("include"), status, None if status != "grounding_failed" else verdict("uncertain", 0))[1] == "pending"


def test_record_refuses_exported_uncertain_and_pending_with_decision():
    base = dict(paper_id="W1", rule=rule(), llm_status="not_run")
    with pytest.raises(ValidationError):
        ScreeningRecord(**base, final="uncertain", origin="agreement")
    with pytest.raises(ValidationError):
        ScreeningRecord(**base, final="include", origin="pending", escalation="x")
    with pytest.raises(ValidationError):
        ScreeningRecord(**base, final="include", origin="human")  # sans auteur ni justification


# -- kappa ---------------------------------------------------------------------
def test_kappa_hand_computed_example():
    pairs = [("include", "include"), ("include", "exclude"), ("exclude", "exclude"), ("exclude", "exclude")]
    s = cohen_kappa(pairs)
    assert s.n == 4 and s.agreed == 3 and s.observed == 0.75 and s.kappa == pytest.approx(0.5)
    assert s.matrix["include|exclude"] == 1


def test_kappa_perfect_agreement_is_one():
    assert cohen_kappa([("include", "include"), ("exclude", "exclude")]).kappa == pytest.approx(1.0)


def test_kappa_total_disagreement_is_negative():
    assert cohen_kappa([("include", "exclude"), ("exclude", "include")]).kappa < 0


def test_kappa_undefined_cases_are_explicit():
    assert cohen_kappa([]).kappa is None and "aucune" in cohen_kappa([]).note
    same = cohen_kappa([("include", "include")] * 3)
    assert same.kappa is None and "indéfini" in same.note and same.observed == 1.0