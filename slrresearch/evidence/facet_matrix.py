"""Step 7: build the study-by-protocol-facet coverage matrix from verified evidence."""
from slrresearch.core.contract import StageResult
from slrresearch.core.errors import ArtifactNotFound
from slrresearch.core.schemas.facet_evidence import FacetEvidenceOutcome
from slrresearch.core.schemas.facet_matrix import (
    FacetMatrixCell, FacetMatrixOutcome, FacetMatrixPaper, FacetMatrixSummary,
)
from slrresearch.core.schemas.facet_plan import FacetPlanOutcome

ARTIFACT = "facet_matrix"
VERSION = "facet-matrix-v1"


def build_facet_matrix(plan: FacetPlanOutcome, evidence: FacetEvidenceOutcome) -> FacetMatrixOutcome:
    if evidence.plan_hash != plan.protocol_sha256:
        raise ValueError("le plan des facettes ne correspond pas à l'artefact de preuves")
    categories = {facet.id: facet for facet in plan.facets}
    matrices = []
    for paper in evidence.papers:
        records = {record.facet_id: record for record in paper.evidence}
        cells = []
        for source in plan.source_facets:
            canonical = categories[source.category]
            record = records.get(source.category)
            source_status = record.status if record else "missing"
            # Only evidence accepted by the independent judge counts as coverage.
            status = source_status if source_status in ("found", "not_reported") else "missing"
            cells.append(FacetMatrixCell(
                paper_id=paper.paper_id, facet_id=source.facet_id, rq_id=source.rq_id,
                category=source.category, blocking=canonical.blocking, status=status,
                evidence_status=source_status,
                value=record.value if record else "", quote=record.quote if record else "",
                chunk_id=record.chunk_id if record else "", page=record.page if record else None,
                note=record.note if record else "Aucune preuve enregistrée pour cette facette.",
            ))
        matrices.append(FacetMatrixPaper(paper_id=paper.paper_id, cells=cells))
    all_cells = [cell for paper in matrices for cell in paper.cells]
    summary = FacetMatrixSummary(
        papers=len(matrices), cells=len(all_cells),
        found=sum(cell.status == "found" for cell in all_cells),
        not_reported=sum(cell.status == "not_reported" for cell in all_cells),
        missing=sum(cell.status == "missing" for cell in all_cells),
        contradictory=sum(cell.status == "contradictory" for cell in all_cells),
        blocking_missing=sum(cell.blocking and cell.status == "missing" for cell in all_cells),
    )
    return FacetMatrixOutcome(version=VERSION, plan_hash=plan.protocol_sha256, papers=matrices, summary=summary)


class FacetMatrixStage:
    name = ARTIFACT

    def run(self, ctx) -> StageResult:
        try:
            plan = ctx.store.read_artifact("facet_plan", FacetPlanOutcome)
            evidence = ctx.store.read_artifact("facet_evidence", FacetEvidenceOutcome)
            matrix = build_facet_matrix(plan, evidence)
        except ArtifactNotFound as exc:
            return StageResult(status="failed", message=f"artefact requis introuvable : {exc}")
        except ValueError as exc:
            return StageResult(status="failed", message=str(exc))

        ctx.store.write_artifact(ARTIFACT, FacetMatrixOutcome, matrix)
        review_items = []
        for paper in matrix.papers:
            for cell in paper.cells:
                if cell.status != "missing" or not cell.blocking:
                    continue
                item = ctx.queue.open_item(
                    "G5", "facet_matrix", f"{cell.paper_id}:{cell.facet_id}",
                    f"Facette bloquante {cell.facet_id} ({cell.category}) manquante pour {cell.paper_id}. "
                    "Confirmer l'absence, lancer une recherche complémentaire ou exclure l'étude de cette synthèse.",
                )
                if item.status == "open":
                    review_items.append(item)
        ctx.trace.log("facet_matrix", "written", **matrix.summary.model_dump())
        if review_items:
            return StageResult(status="waiting_human", artifact=ARTIFACT, review_items=review_items,
                               message=f"{len(review_items)} facettes bloquantes manquantes attendent une revue G5")
        return StageResult(status="done", artifact=ARTIFACT,
                           message="Matrice étude × facette construite depuis les preuves vérifiées.")
