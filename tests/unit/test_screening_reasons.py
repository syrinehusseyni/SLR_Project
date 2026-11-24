from slrresearch.screening.stage import ScreeningStage
from tests.screening_helpers import TEMPLATE, make_paper, setup_run


def test_review_items_show_the_current_reason_not_a_stale_one(store):
    ctx, _ = setup_run(store, [make_paper("NOABS", abstract="")])
    ctx.queue.open_item("G2", "paper", "NOABS", "ancienne raison périmée d'un run précédent")
    r = ScreeningStage(prompt_template=TEMPLATE).run(ctx)
    assert r.status == "waiting_human"
    shown = r.review_items[0].reason
    assert "Résumé absent" in shown and "périmée" not in shown