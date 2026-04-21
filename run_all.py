"""
run_all.py
==========
Vérifie tous les fichiers src/ puis exécute tous les notebooks dans l'ordre.

Usage :
    cd anomaly_detection_project
    python run_all.py                   # tout exécuter
    python run_all.py --from 03         # reprendre à partir du NB03
    python run_all.py --only 04         # un seul notebook
    python run_all.py --check-only      # vérifier src/ sans lancer les notebooks
    python run_all.py --timeout 3600    # timeout par notebook (défaut: 1200s)

Prérequis :
    pip install nbconvert nbformat jupyter
"""

import subprocess, sys, time, argparse, importlib
from pathlib import Path

GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
BLUE   = "\033[94m"; BOLD   = "\033[1m";  RESET  = "\033[0m"

# ── Notebooks dans l'ordre ─────────────────────────────────────────────────────
NOTEBOOKS = [
    ("01", "notebooks/01_data_understanding.ipynb"),
    ("02", "notebooks/02_data_preparation.ipynb"),
    ("03", "notebooks/03_baseline_models.ipynb"),
    ("04", "notebooks/04_autoencoder.ipynb"),
]

# ── Fichiers src/ à valider + quel(s) notebook(s) les utilise(nt) ─────────────
SRC_MODULES = [
    ("src.feature_engineering.feature_builder", "NB02"),
    ("src.preprocessing.preprocessing",          "NB02"),
    ("src.visualization.prep_plots",            "NB02"),
    ("src.models.ml_models",                    "NB03"),
    ("src.utils.evaluator",                     "NB03 + NB04"),
    ("src.utils.baseline_config",               "NB03 + NB04"),
    ("src.visualization.model_plots",           "NB03"),
    ("src.models.autoencoder",                  "NB04"),
    ("src.visualization.autoencoder_plots",     "NB04"),
]

def check_src_modules() -> bool:
    """Importe chaque module src/ et signale les erreurs."""
    print(f"{BOLD}── Validation des fichiers src/ ──────────────────────{RESET}")
    all_ok = True
    for module, used_by in SRC_MODULES:
        try:
            importlib.import_module(module)
            print(f"  {GREEN}✅{RESET}  {module:<45} (utilisé par {used_by})")
        except Exception as e:
            print(f"  {RED}❌{RESET}  {module:<45} → {RED}{type(e).__name__}: {e}{RESET}")
            all_ok = False
    print()
    return all_ok

def run_notebook(nb_path: Path, timeout: int) -> tuple[bool, float, str]:
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = round(time.time() - t0, 1)
    if result.returncode != 0:
        err = result.stderr[-800:] if result.stderr else result.stdout[-800:]
        return False, duration, err
    return True, duration, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",       dest="from_nb",    default=None)
    parser.add_argument("--only",       dest="only_nb",    default=None)
    parser.add_argument("--timeout",    dest="timeout",    default=1200, type=int)
    parser.add_argument("--check-only", dest="check_only", action="store_true",
                        help="Valider src/ uniquement, sans lancer les notebooks")
    args = parser.parse_args()

    # Ajouter project root au path
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  Projet Détection de Fraudes — Exécution complète{RESET}")
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"  Répertoire : {project_root}\n")

    # ── Étape 1 : validation src/ ──────────────────────────────────────────────
    src_ok = check_src_modules()

    if not src_ok:
        print(f"{RED}❌ Erreurs dans src/ — corriger avant de lancer les notebooks.{RESET}\n")
        sys.exit(1)

    print(f"{GREEN}✅ Tous les modules src/ sont valides.{RESET}\n")

    if args.check_only:
        print("Mode --check-only : notebooks non exécutés.\n")
        return

    # ── Étape 2 : sélection des notebooks ─────────────────────────────────────
    to_run = NOTEBOOKS
    if args.only_nb:
        to_run = [(id_, p) for id_, p in NOTEBOOKS if id_ == args.only_nb]
    elif args.from_nb:
        ids   = [id_ for id_, _ in NOTEBOOKS]
        start = ids.index(args.from_nb) if args.from_nb in ids else 0
        to_run = NOTEBOOKS[start:]

    print(f"{BOLD}── Exécution des notebooks ({len(to_run)}) ────────────────────{RESET}")
    print(f"  Timeout par notebook : {args.timeout}s\n")

    # ── Étape 3 : exécution ───────────────────────────────────────────────────
    results = []
    total_start = time.time()

    for id_, nb_rel in to_run:
        nb_path = Path(nb_rel)
        if not nb_path.exists():
            print(f"{YELLOW}⚠  NB{id_} — {nb_path.name} introuvable, ignoré.{RESET}\n")
            results.append((id_, nb_path.name, "SKIP", 0))
            continue

        print(f"{BLUE}▶  NB{id_} — {nb_path.name}{RESET}  ", end="", flush=True)
        ok, duration, err = run_notebook(nb_path, args.timeout)

        if ok:
            print(f"{GREEN}✅ OK  ({duration}s){RESET}")
            results.append((id_, nb_path.name, "OK", duration))
        else:
            print(f"{RED}❌ ERREUR  ({duration}s){RESET}")
            for line in err.strip().splitlines()[-8:]:
                print(f"   {line}")
            results.append((id_, nb_path.name, "FAIL", duration))
            print(f"\n{YELLOW}⚠  Arrêt — corriger NB{id_} avant de continuer.{RESET}\n")
            break
        print()

    # ── Résumé ─────────────────────────────────────────────────────────────────
    total = round(time.time() - total_start, 1)
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  RÉSUMÉ FINAL{RESET}")
    print(f"{BOLD}{'='*58}{RESET}")
    for id_, name, status, dur in results:
        icon  = "✅" if status == "OK" else ("⚠ " if status == "SKIP" else "❌")
        color = GREEN if status == "OK" else (YELLOW if status == "SKIP" else RED)
        print(f"  {icon}  NB{id_}  {name:<42} {color}{status}{RESET}  {dur}s")

    n_ok = sum(1 for *_, s, __ in results if s == "OK")
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"  {n_ok}/{len(results)} notebooks réussis — {total}s total\n")

    if any(s == "FAIL" for *_, s, __ in results):
        sys.exit(1)

if __name__ == "__main__":
    main()