"""Ligne de commande minimale (PowerShell) pour dérouler les étapes déjà écrites.

  python cli.py questions --run-id demo --topic "Utilisation des LLM pour ..."
  python cli.py protocol  --run-id demo --year-from 2018
  python cli.py gates     --run-id demo
  python cli.py close     --run-id demo --item G1:protocol:main --decision approved --by moi --why "relu, OK"
  python cli.py protocol  --run-id demo        # reprise : G1 fermée -> done
  python cli.py search    --run-id demo        # étape 2 : interroge OpenAlex, Crossref, Semantic Scholar, arXiv
  python cli.py screening --run-id demo        # étape 3 : screening A (règles) + B (LLM) + porte G2
  python cli.py screening --run-id demo --pause 3 --batch 50   # Groq : 3 s entre appels, 50 papiers par lot
  python cli.py screening --run-id demo --max-papers 60        # refuse de démarrer si plus de 60 papiers pour le LLM
  python cli.py close     --run-id demo --item "G2:paper:W123" --decision include --by moi --why "..."
  python cli.py download  --run-id demo        # étape 4 : PDF des papiers inclus ; sans PDF libre => indisponible (automatique)
  python cli.py download  --run-id demo --on-missing review   # variante : un point G3 par papier sans PDF (dépôt manuel)
  python cli.py download  --run-id demo --max-papers 25 --max-mb 30 --pause 2 --batch 10   # limites d'exécution
  python cli.py close     --run-id demo --item "G3:paper:W123" --decision provided --by moi --why "PDF déposé"
  python cli.py close     --run-id demo --item "G3:paper:W123" --decision unavailable --by moi --why "introuvable"
  python cli.py extraction --run-id demo       # étape 5a : lit les PDF, nettoie, découpe en chunks (sans LLM)
  python cli.py extraction --run-id demo --max-pages 80 --chunk-size 1200 --chunk-overlap 200 --max-papers 30 --force
  python cli.py studycards --run-id demo       # étape 5b : fiche d'étude par LLM (5 champs par papier), citations vérifiées
  python cli.py studycards --run-id demo --pause 3 --max-calls 120 --max-tokens 400000   # Groq : 3 s entre appels, budgets
  python cli.py studycards --run-id demo --max-papers 20 --top-k 3 --force

Codes de sortie : 0 = étape terminée, 2 = en attente d'une décision humaine, 1 = échec.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from slrresearch.core.budgets import Budget
from slrresearch.core.config import build_model, load_settings
from slrresearch.core.contract import build_context
from slrresearch.core.errors import SLRError
from slrresearch.core.schemas.documents import DocumentsOutcome
from slrresearch.core.schemas.extraction import ExtractionOutcome
from slrresearch.core.schemas.screening import ScreeningOutcome
from slrresearch.core.schemas.search import SearchOutcome
from slrresearch.core.schemas.study_card import StudyCardsOutcome
from slrresearch.core.schemas.fidelity import FidelityOutcome
from slrresearch.core.store import RunStore
from slrresearch.download.agent import DownloadSettings
from slrresearch.download.stage import G3_DECISIONS, MISSING_POLICIES, POLICY_RESOLVER, DownloadStage, manual_pdf_problem
from slrresearch.extraction.card_stage import DEFAULT_TOP_K, StudyCardStage
from slrresearch.extraction.stage import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, ExtractionStage
from slrresearch.evidence.fidelity import FidelityStage
from slrresearch.evidence.facet_plan import FacetPlanStage
from slrresearch.evidence.facet_evidence import FacetEvidenceStage
from slrresearch.core.schemas.facet_evidence import FacetEvidenceOutcome
from slrresearch.evidence.facet_matrix import FacetMatrixStage
from slrresearch.core.schemas.facet_matrix import FacetMatrixOutcome
from slrresearch.protocol.questions_stage import QuestionsStage
from slrresearch.protocol.stage import ProtocolStage
from slrresearch.screening.stage import HUMAN_DECISIONS, ScreeningStage
from slrresearch.search.sources import SearchConfig
from slrresearch.search.stage import SearchStage

GATES = ("G1", "G2", "G3", "G4", "G5", "G6")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    parser.add_argument("--root", default="runs", help="dossier des runs (défaut : runs)")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("questions", help="étape 1a : générer les questions de recherche")
    q.add_argument("--run-id", required=True)
    q.add_argument("--topic", required=True)
    q.add_argument("--n", type=int, default=3)

    p = sub.add_parser("protocol", help="étape 1b : protocole complet + porte G1")
    p.add_argument("--run-id", required=True)
    p.add_argument("--year-from", type=int, default=None)

    s = sub.add_parser("search", help="étape 2 : exécuter le plan de recherche")
    s.add_argument("--run-id", required=True)
    s.add_argument("--limit", type=int, default=50, help="résultats max par requête (défaut 50)")
    s.add_argument("--force", action="store_true", help="refaire la recherche même si l'artefact existe")

    sc = sub.add_parser("screening", help="étape 3 : screening titre/résumé + porte G2")
    sc.add_argument("--run-id", required=True)
    sc.add_argument("--min-confidence", type=float, default=0.6, help="seuil de confiance du LLM (défaut 0.6)")
    sc.add_argument("--max-calls", type=int, default=600, help="budget d'appels LLM (défaut 600)")
    sc.add_argument("--force", action="store_true", help="recalculer les règles et l'arbitrage (le cache LLM est gardé)")
    sc.add_argument("--batch", type=int, default=None, help="traiter au plus N nouveaux papiers par le LLM, puis s'arrêter")
    sc.add_argument("--pause", type=float, default=0.0, help="secondes d'attente entre deux appels LLM (ex. 3 pour Groq)")
    sc.add_argument("--max-papers", type=int, default=None,
                    help="refuser de démarrer si plus de N papiers doivent passer par le LLM (aucun appel n'est fait)")
    sc.add_argument("--retries", type=int, default=2, help="nouveaux essais d'un papier dont l'appel LLM échoue (défaut 2)")
    sc.add_argument("--retry-wait", type=float, default=20.0, help="secondes avant le 1er nouvel essai (défaut 20, puis 40...)")

    d = sub.add_parser("download", help="étape 4 : télécharger les PDF des papiers inclus + porte G3")
    d.add_argument("--run-id", required=True)
    d.add_argument("--max-papers", type=int, default=None,
                   help="refuser de démarrer si plus de N papiers doivent être téléchargés (aucun téléchargement n'est fait)")
    d.add_argument("--max-mb", type=float, default=50.0, help="taille maximale d'un PDF en Mo (défaut 50)")
    d.add_argument("--pause", type=float, default=1.0, help="secondes entre deux téléchargements (défaut 1 ; arXiv : 3 imposées)")
    d.add_argument("--batch", type=int, default=None, help="télécharger au plus N papiers par lancement, puis s'arrêter")
    d.add_argument("--max-calls", type=int, default=400, help="budget d'appels HTTP du lancement (défaut 400)")
    d.add_argument("--min-title-coverage", type=float, default=0.7,
                   help="part des mots du titre à retrouver dans le PDF (défaut 0.7)")
    d.add_argument("--on-missing", choices=MISSING_POLICIES, default="unavailable",
                   help="papier sans PDF en accès libre : « unavailable » (défaut) le déclare indisponible automatiquement ; "
                        "« review » ouvre un point G3 (dépôt manuel du PDF)")
    d.add_argument("--retry-manual", action="store_true",
                   help="retenter les papiers sans PDF et sans décision humaine (y compris ceux déclarés indisponibles automatiquement)")
    d.add_argument("--force", action="store_true", help="ignorer l'état précédent et tout retélécharger")

    e = sub.add_parser("extraction", help="étape 5a : lire les PDF téléchargés et les découper en chunks (sans LLM)")
    e.add_argument("--run-id", required=True)
    e.add_argument("--max-papers", type=int, default=None,
                   help="refuser de démarrer si plus de N papiers doivent être extraits (aucune extraction n'est faite)")
    e.add_argument("--max-pages", type=int, default=None, help="lire au plus N pages par papier (défaut : toutes)")
    e.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"taille d'un chunk en caractères (défaut {DEFAULT_CHUNK_SIZE})")
    e.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                   help=f"chevauchement entre deux chunks en caractères (défaut {DEFAULT_CHUNK_OVERLAP})")
    e.add_argument("--force", action="store_true", help="ignorer les papiers déjà extraits et tout refaire")

    st = sub.add_parser("studycards", help="étape 5b : fiche d'étude par LLM, citations vérifiées")
    st.add_argument("--run-id", required=True)
    st.add_argument("--max-calls", type=int, default=100, help="budget d'appels LLM (défaut 100 : 5 champs par papier + réparations)")
    st.add_argument("--max-tokens", type=int, default=None, help="budget de tokens (défaut : aucun)")
    st.add_argument("--max-seconds", type=float, default=None, help="budget de temps en secondes (défaut : aucun)")
    st.add_argument("--max-papers", type=int, default=None,
                    help="refuser de démarrer si plus de N papiers doivent passer par le LLM (aucun appel n'est fait)")
    st.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"nombre de chunks montrés au LLM par champ (défaut {DEFAULT_TOP_K})")
    st.add_argument("--pause", type=float, default=0.0, help="secondes d'attente entre deux appels LLM (ex. 3 pour Groq)")
    st.add_argument("--force", action="store_true", help="refaire toutes les fiches (sinon seuls les champs manquants ou en panne sont refaits)")

    fj = sub.add_parser("fidelity", help="étape 6a : juge indépendant des citations de fiches d'étude")
    fj.add_argument("--run-id", required=True)
    fj.add_argument("--max-calls", type=int, default=60, help="budget d'appels du juge (défaut 60)")
    fj.add_argument("--max-tokens", type=int, default=None)
    fj.add_argument("--max-seconds", type=float, default=None)
    fj.add_argument("--max-papers", type=int, default=None)
    fj.add_argument("--pause", type=float, default=0.0)
    fj.add_argument("--force", action="store_true", help="ignorer le cache et refaire les jugements")

    fp = sub.add_parser("facet-plan", help="étape 6b : préparer et faire valider le plan normalisé des facettes")
    fp.add_argument("--run-id", required=True)

    fe = sub.add_parser("facet-evidence", help="étape 6b : extraire et vérifier les preuves par facette")
    fe.add_argument("--run-id", required=True)
    fe.add_argument("--max-calls", type=int, default=100)
    fe.add_argument("--max-tokens", type=int, default=None)
    fe.add_argument("--max-seconds", type=float, default=None)
    fe.add_argument("--max-papers", type=int, default=None)
    fe.add_argument("--top-k", type=int, default=3)
    fe.add_argument("--pause", type=float, default=2.1)
    fe.add_argument("--force", action="store_true", help="ignorer les résultats supplémentaires en cache")
    fe.add_argument("--repair-missing", action="store_true",
                    help="relancer extraction et juge pour chaque facette sans preuve acceptée")

    fm = sub.add_parser("facet-matrix", help="étape 7 : matrice étude × facette et revue G5 des manques bloquants")
    fm.add_argument("--run-id", required=True)

    g = sub.add_parser("gates", help="lister les points de revue ouverts")
    g.add_argument("--run-id", required=True)

    c = sub.add_parser("close", help="fermer un point de revue avec justification")
    c.add_argument("--run-id", required=True)
    c.add_argument("--item", required=True)
    c.add_argument("--decision", required=True, help="approved pour valider ; autre valeur = refus")
    c.add_argument("--by", required=True)
    c.add_argument("--why", required=True)
    return parser


def _report(result) -> int:
    if result.status == "done":
        print(f"OK : étape terminée, artefact « {result.artifact} ».")
        return 0
    if result.status == "waiting_human":
        print(f"EN ATTENTE : artefact « {result.artifact} » écrit, décision humaine requise :")
        for item in result.review_items:
            print(f"  - {item.id} : {item.reason}")
        print("Relisez le fichier, puis : python cli.py close --run-id ... --item <id> --decision approved --by ... --why ...")
        return 2
    print(f"ÉCHEC : {result.message}")
    return 1


def _print_search_summary(store) -> None:
    outcome = store.read_artifact("search", SearchOutcome)
    summary = outcome.summary
    print(f"Notices récupérées : {summary.retrieved} ({', '.join(f'{k}: {v}' for k, v in sorted(summary.per_source.items()))})")
    print(f"Doublons retirés : {summary.duplicates_removed} ; papiers uniques : {summary.unique} ; sans résumé : {summary.without_abstract}")
    for run in outcome.runs:
        if run.status != "ok":
            print(f"  ! {run.source} [{run.status}] : {run.error}")


def _print_screening_summary(store) -> None:
    outcome = store.read_artifact("screening", ScreeningOutcome)
    s = outcome.summary
    print(f"Papiers : {s.total} ; inclus : {s.included} ; exclus : {s.excluded} ; en attente (G2) : {s.pending}")
    print(f"Origine des décisions : {dict(sorted(s.by_origin.items()))} ; sans résumé : {s.without_abstract} ; échecs LLM : {s.llm_failures}")
    a = s.agreement
    if a.n:
        kappa = f"{a.kappa:.3f}" if a.kappa is not None else f"indéfini ({a.note})"
        print(f"Accord A/B sur {a.n} papiers : {a.agreed} identiques ({a.observed:.0%}) ; kappa de Cohen : {kappa}")
    else:
        print("Accord A/B : aucune paire comparable.")


def _print_documents_summary(store) -> None:
    outcome = store.read_artifact("documents", DocumentsOutcome)
    s, lim = outcome.summary, outcome.limits
    by_policy = sum(1 for r in outcome.records
                    if r.status == "unavailable" and r.attempts and r.attempts[-1].resolver == POLICY_RESOLVER)
    print(f"Papiers inclus : {s.total} ; PDF obtenus : {s.downloaded} ; à fournir à la main (G3) : {s.manual_required} ; "
          f"indisponibles : {s.unavailable}")
    if by_policy:
        print(f"Dont {by_policy} déclarés indisponibles automatiquement (aucun PDF en accès libre téléchargeable) : "
              "à compter comme « rapports non obtenus » dans le rapport.")
    if s.by_resolver:
        print("PDF obtenus par : " + ", ".join(f"{k}: {v}" for k, v in sorted(s.by_resolver.items())))
    print(f"Limites de ce run : on-missing={lim.on_missing} ; max-papers={lim.max_papers} ; max-mb={lim.max_mb:g} ; max-calls={lim.max_http_calls} ; "
          f"pause={lim.pause:g}s ; couverture du titre >= {lim.min_title_coverage:g}")


def _print_extraction_summary(store) -> None:
    outcome = store.read_artifact("extraction", ExtractionOutcome)
    s, lim = outcome.summary, outcome.limits
    print(f"Papiers téléchargés : {s.total} ; extraits : {s.extracted} ; en échec : {s.failed} ; chunks : {s.chunks} ; "
          f"sans PDF (étape 4) : {s.skipped_unavailable}")
    for p in outcome.papers:
        if p.status == "failed":
            print(f"  ! {p.paper_id} : {p.reason}")
        elif p.truncated:
            print(f"  ~ {p.paper_id} : tronqué ({p.pages_read}/{p.total_pages} pages lues)")
    pages = lim.max_pages if lim.max_pages is not None else "toutes"
    print(f"Réglages de ce run : chunks de {lim.chunk_size} caractères, chevauchement {lim.chunk_overlap} ; pages lues : {pages} ; "
          f"max-papers={lim.max_papers}")


def _print_cards_summary(store) -> None:
    outcome = store.read_artifact("study_cards", StudyCardsOutcome)
    s, lim = outcome.summary, outcome.limits
    print(f"Papiers : {s.total} ; fiches complètes : {s.complete} ; incomplètes : {s.incomplete} ; en échec : {s.failed} ; "
          f"non extraits (5a) : {s.skipped_not_extracted}")
    print(f"Champs par statut : {dict(sorted(s.fields.items()))}")
    print(f"Coût de ce lancement : {s.llm_calls} appels LLM, {s.tokens} tokens ; modèle : {lim.model} ; top-k : {lim.top_k}")
    for p in outcome.papers:
        if p.status == "incomplete":
            print(f"  ~ {p.paper_id} : champs en panne LLM, relancez la même commande pour les refaire")
        elif p.status == "failed":
            print(f"  ! {p.paper_id} : {p.reason}")
    if s.fields.get("rejected"):
        print("Les champs « rejected » (citation absente du chunk ou nombre inexact) ne sont pas des faits : ne les utilisez pas.")


def _print_fidelity_summary(store) -> None:
    outcome = store.read_artifact("fidelity", FidelityOutcome)
    s = outcome.summary
    print(f"Papiers : {s.papers} ; vérifiés : {s.verified} ; revue G5 : {s.human_review} ; rejetés : {s.rejected} ; erreurs : {s.llm_error} ; ignorés : {s.skipped}")
    print(f"Coût de ce lancement : {s.llm_calls} appels, {s.tokens} tokens ; juge : {outcome.judge_model}")


def _print_facet_evidence_summary(store) -> None:
    outcome = store.read_artifact("facet_evidence", FacetEvidenceOutcome)
    s = outcome.summary
    print(f"Papiers : {s.papers} ; preuves : {s.found} ; absentes : {s.not_reported} ; à chercher : {s.missing} ; "
          f"revue G5 : {s.human_review} ; rejetées : {s.rejected} ; erreurs : {s.llm_error}")
    print(f"Réutilisées depuis l'étape 5b/6a : {s.reused} ; coût de ce lancement : {s.llm_calls} appels, {s.tokens} tokens")


def _print_facet_matrix_summary(store) -> None:
    s = store.read_artifact("facet_matrix", FacetMatrixOutcome).summary
    print(f"Papiers : {s.papers} ; cellules : {s.cells} ; trouvées : {s.found} ; non rapportées : {s.not_reported}")
    print(f"Manquantes : {s.missing} (bloquantes : {s.blocking_missing}) ; contradictoires : {s.contradictory}")


def main(argv=None, make_model=build_model, transport=None, env=None, stage_options=None) -> int:
    args = _parser().parse_args(argv)
    store = RunStore(Path(args.root), args.run_id)
    try:
        settings = load_settings()
        budget = Budget(max_calls=getattr(args, "max_calls", 40), max_seconds=getattr(args, "max_seconds", None),
                        max_tokens=getattr(args, "max_tokens", None))
        ctx = build_context(store, settings, budget=budget, make_model=make_model)

        if args.command == "questions":
            return _report(QuestionsStage(args.topic, args.n).run(ctx))
        if args.command == "protocol":
            return _report(ProtocolStage(year_from=args.year_from).run(ctx))
        if args.command == "search":
            if env is None:  # variables d'environnement prioritaires sur le fichier .env
                env = {**dotenv_values(".env"), **os.environ}
            stage = SearchStage(SearchConfig.from_env(env), results_per_query=args.limit, force=args.force,
                                transport=transport, **(stage_options or {}))
            code = _report(stage.run(ctx))
            if code == 0:
                _print_search_summary(store)
            return code
        if args.command == "screening":
            def show(index, total, paper_id, status):
                print(f"[{index}/{total}] {paper_id} : {status}", flush=True)

            stage = ScreeningStage(min_confidence=args.min_confidence, force=args.force, batch=args.batch,
                                   pause=args.pause, max_papers=args.max_papers, retries=args.retries,
                                   retry_wait=args.retry_wait, progress=show)
            code = _report(stage.run(ctx))
            if store.exists("screening"):
                _print_screening_summary(store)
            return code
        if args.command == "download":
            if env is None:
                env = {**dotenv_values(".env"), **os.environ}
            config = SearchConfig.from_env(env)
            if not config.contact_email:
                print("NOTE : CONTACT_EMAIL absent du .env : Unpaywall (qui exige un e-mail) sera ignoré.")

            def show_download(index, total, paper_id, status):
                print(f"[{index}/{total}] {paper_id} : {status}", flush=True)

            settings_dl = DownloadSettings(max_bytes=int(args.max_mb * 1024 * 1024), min_title_coverage=args.min_title_coverage)
            stage = DownloadStage(config, settings_dl, max_papers=args.max_papers, pause=args.pause, batch=args.batch,
                                  max_http_calls=args.max_calls, force=args.force, retry_manual=args.retry_manual,
                                  on_missing=args.on_missing, progress=show_download, transport=transport, **(stage_options or {}))
            try:
                code = _report(stage.run(ctx))
            except KeyboardInterrupt:
                print("INTERROMPU : la progression des téléchargements est sauvegardée, relancez la même commande pour reprendre.")
                return 130
            if code == 2:
                print("Pour chaque point G3 : déposez le PDF dans runs\\<run-id>\\documents\\manual\\<paper_id>.pdf puis fermez avec "
                      "--decision provided, ou fermez avec --decision unavailable si vous ne le trouvez pas.")
            if store.exists("documents"):
                _print_documents_summary(store)
            return code
        if args.command == "extraction":
            def show_extraction(index, total, paper_id, status):
                print(f"[{index}/{total}] {paper_id} : {status}", flush=True)

            try:
                stage = ExtractionStage(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, max_pages=args.max_pages,
                                        max_papers=args.max_papers, force=args.force, progress=show_extraction)
            except ValueError as exc:  # ex. chevauchement >= taille
                print(f"ERREUR : {exc}")
                return 1
            try:
                code = _report(stage.run(ctx))
            except KeyboardInterrupt:
                print("INTERROMPU : les papiers déjà extraits sont conservés, relancez la même commande pour reprendre.")
                return 130
            if code == 0:
                _print_extraction_summary(store)
            return code
        if args.command == "studycards":
            def show_card(index, total, paper_id, status):
                print(f"[{index}/{total}] {paper_id} : {status}", flush=True)

            try:
                stage = StudyCardStage(top_k=args.top_k, max_papers=args.max_papers, pause=args.pause, force=args.force,
                                       progress=show_card)
            except ValueError as exc:
                print(f"ERREUR : {exc}")
                return 1
            try:
                code = _report(stage.run(ctx))
            except KeyboardInterrupt:
                print("INTERROMPU : les fiches déjà faites sont conservées, relancez la même commande pour reprendre.")
                return 130
            if code == 0:
                _print_cards_summary(store)
            return code
        if args.command == "fidelity":
            if not settings.judge_is_independent:
                print(f"ERREUR : le juge ({settings.model_for('judge')}) doit différer de l'extracteur ({settings.model_for('extraction')}). Configurez le modèle juge dans .env.")
                return 1
            def show_fidelity(index, total, paper_id, status):
                print(f"[{index}/{total}] {paper_id} : {status}", flush=True)
            stage = FidelityStage(max_papers=args.max_papers, pause=args.pause, force=args.force)
            try:
                code = _report(stage.run(ctx))
            except KeyboardInterrupt:
                print("INTERROMPU : la trace est conservée ; relancez la même commande pour reprendre.")
                return 130
            if store.exists("fidelity"):
                _print_fidelity_summary(store)
            return code
        if args.command == "facet-plan":
            return _report(FacetPlanStage().run(ctx))
        if args.command == "facet-evidence":
            try:
                stage = FacetEvidenceStage(top_k=args.top_k, max_papers=args.max_papers, pause=args.pause,
                                           force=args.force, repair_missing=args.repair_missing)
            except ValueError as exc:
                print(f"ERREUR : {exc}")
                return 1
            code = _report(stage.run(ctx))
            if store.exists("facet_evidence"):
                _print_facet_evidence_summary(store)
            return code
        if args.command == "facet-matrix":
            code = _report(FacetMatrixStage().run(ctx))
            if store.exists("facet_matrix"):
                _print_facet_matrix_summary(store)
            return code
        if args.command == "gates":
            found = False
            for gate in GATES:
                for item in ctx.queue.open_items(gate):
                    found = True
                    print(f"{item.id} [{item.gate}] : {item.reason}")
            if not found:
                print("Aucun point de revue ouvert.")
            return 0
        if args.command == "close":
            decision = args.decision
            if args.item.startswith("G2:"):
                decision = decision.strip().lower()
                if decision not in HUMAN_DECISIONS:  # un point fermé ne peut plus être rouvert : on refuse avant
                    print(f"ERREUR : pour la porte G2, la décision doit être « include » ou « exclude » (reçu : {args.decision!r}).")
                    return 1
            if args.item.startswith("G3:"):
                decision = decision.strip().lower()
                if decision not in G3_DECISIONS:  # un point fermé ne peut plus être rouvert : on refuse avant
                    print(f"ERREUR : pour la porte G3, la décision doit être « provided » ou « unavailable » (reçu : {args.decision!r}).")
                    return 1
                if decision == "provided":
                    paper_id = args.item.split(":", 2)[2]
                    paper = next((p for p in store.read_artifact("search", SearchOutcome).papers if p.id == paper_id), None)
                    if paper is None:
                        print(f"ERREUR : papier inconnu dans l'artefact search : {paper_id}")
                        return 1
                    problem = manual_pdf_problem(store, paper, DownloadSettings())
                    if problem:
                        print(f"ERREUR : le PDF déposé est inutilisable : {problem}")
                        return 1
            if args.item.startswith("G5:"):
                decision = decision.strip().lower()
                if decision not in ("approved", "rejected"):
                    print(f"ERREUR : pour la porte G5, la décision doit être « approved » ou « rejected » (reçu : {args.decision!r}).")
                    return 1
            item = ctx.queue.close_item(args.item, decision, args.by, args.why)
            print(f"Fermé : {item.id} -> {item.decision} par {item.decided_by}.")
            return 0
    except SLRError as exc:
        print(f"ERREUR : {exc}")
        return 1
    except KeyboardInterrupt:
        print("INTERROMPU : la progression LLM est sauvegardée, relancez la même commande pour reprendre.")
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
