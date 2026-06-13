# -*- coding: utf-8 -*-
"""
src/monitoring/drift_detector.py
==================================
Détection de dérive des données (data drift) pour surveiller
la stabilité du modèle en production.

Méthode principale : PSI (Population Stability Index)
  PSI < 0.10  → stable (pas de dérive)
  PSI 0.10-0.25 → dérive modérée (à surveiller)
  PSI > 0.25  → dérive critique (réentraînement recommandé)

Usage :
    detector = DriftDetector(X_train_ref, feature_names)
    rapport  = detector.compute_psi_report(X_new)
    detector.print_report(rapport)
    detector.save_report(rapport, "outputs/audit_trail/drift_report.json")
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Seuils PSI ────────────────────────────────────────────────────────────────

PSI_STABLE   = 0.10
PSI_MODERATE = 0.25


def _psi_single(
    ref_vals : np.ndarray,
    new_vals : np.ndarray,
    n_bins   : int = 10,
    epsilon  : float = 1e-6,
) -> float:
    """
    Calcule le PSI entre une distribution de référence et une nouvelle.

    PSI = Σ (P_new - P_ref) × ln(P_new / P_ref)
    """
    # Pour les variables binaires (0/1), on utilise 2 bins
    unique = np.unique(np.concatenate([ref_vals, new_vals]))
    if len(unique) <= 2:
        bins = np.array([-0.5, 0.5, 1.5])
    else:
        # Quantiles de référence pour des bins équilibrés
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.unique(np.percentile(ref_vals, quantiles))
        if len(bins) < 2:
            return 0.0

    ref_counts, _ = np.histogram(ref_vals, bins=bins)
    new_counts, _ = np.histogram(new_vals, bins=bins)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + epsilon
    new_pct = new_counts / max(new_counts.sum(), 1) + epsilon

    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
    return round(psi, 6)


# ── Classe principale ──────────────────────────────────────────────────────────

class DriftDetector:
    """
    Détecte la dérive des données par rapport à un dataset de référence.

    Parameters
    ----------
    X_reference : np.ndarray | pd.DataFrame
        Dataset d'entraînement (ou validation) servant de référence.
    feature_names : list[str]
        Noms des features dans l'ordre des colonnes de X_reference.
    n_bins : int
        Nombre de bins pour le calcul PSI des variables continues (défaut 10).
    """

    def __init__(
        self,
        X_reference  : np.ndarray | pd.DataFrame,
        feature_names: list[str],
        n_bins       : int = 10,
    ):
        if isinstance(X_reference, pd.DataFrame):
            X_reference = X_reference.values
        self.X_ref         = X_reference.astype(float)
        self.feature_names = feature_names
        self.n_bins        = n_bins
        self.ref_stats     = self._compute_stats(self.X_ref)

    def _compute_stats(self, X: np.ndarray) -> dict:
        return {
            "mean" : X.mean(axis=0).tolist(),
            "std"  : X.std(axis=0).tolist(),
            "min"  : X.min(axis=0).tolist(),
            "max"  : X.max(axis=0).tolist(),
            "p25"  : np.percentile(X, 25, axis=0).tolist(),
            "p50"  : np.percentile(X, 50, axis=0).tolist(),
            "p75"  : np.percentile(X, 75, axis=0).tolist(),
        }

    def compute_psi_report(
        self,
        X_new        : np.ndarray | pd.DataFrame,
        batch_name   : str = "production_batch",
    ) -> dict:
        """
        Calcule le rapport PSI complet entre X_ref et X_new.

        Returns
        -------
        dict avec : timestamp, batch_name, global_psi, status,
                    feature_psi (détail par feature), new_stats.
        """
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values
        X_new = X_new.astype(float)

        feature_psi = {}
        for i, name in enumerate(self.feature_names):
            psi = _psi_single(
                self.X_ref[:, i],
                X_new[:, i],
                n_bins=self.n_bins,
            )
            if psi < PSI_STABLE:
                status = "stable"
            elif psi < PSI_MODERATE:
                status = "moderate_drift"
            else:
                status = "critical_drift"
            feature_psi[name] = {"psi": psi, "status": status}

        psi_values  = [v["psi"] for v in feature_psi.values()]
        global_psi  = round(float(np.mean(psi_values)), 6)
        max_psi     = round(float(np.max(psi_values)), 6)

        if max_psi < PSI_STABLE:
            global_status = "stable"
        elif max_psi < PSI_MODERATE:
            global_status = "moderate_drift"
        else:
            global_status = "critical_drift"

        new_stats = self._compute_stats(X_new)

        return {
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "batch_name"     : batch_name,
            "n_ref"          : int(len(self.X_ref)),
            "n_new"          : int(len(X_new)),
            "global_psi_mean": global_psi,
            "global_psi_max" : max_psi,
            "status"         : global_status,
            "alert"          : global_status == "critical_drift",
            "feature_psi"    : feature_psi,
            "ref_stats"      : self.ref_stats,
            "new_stats"      : new_stats,
        }

    def print_report(self, report: dict) -> None:
        """Affiche un résumé du rapport PSI dans le terminal."""
        sep = "=" * 62
        status_icon = {"stable": "✅", "moderate_drift": "⚠️", "critical_drift": "🚨"}
        icon = status_icon.get(report["status"], "?")

        print(sep)
        print(f"  RAPPORT DE DÉRIVE DES DONNÉES  {icon} {report['status'].upper()}")
        print(f"  {report['timestamp']}")
        print(sep)
        print(f"  Référence : {report['n_ref']:,} transactions  →  Nouveau : {report['n_new']:,}")
        print(f"  PSI moyen : {report['global_psi_mean']:.4f}  |  PSI max : {report['global_psi_max']:.4f}")
        print(f"  Seuils    : stable < {PSI_STABLE} | modéré < {PSI_MODERATE} | critique ≥ {PSI_MODERATE}")
        print("-" * 62)
        print(f"  {'Feature':<30} {'PSI':>8}  {'Statut'}")
        print("-" * 62)
        for name, vals in sorted(
            report["feature_psi"].items(), key=lambda x: x[1]["psi"], reverse=True
        ):
            icon_f = status_icon.get(vals["status"], "?")
            print(f"  {name:<30} {vals['psi']:>8.4f}  {icon_f} {vals['status']}")
        print(sep)
        if report["alert"]:
            print("  ⚠️  ALERTE : dérive critique détectée — réentraînement recommandé.")
            print(sep)

    def save_report(
        self,
        report    : dict,
        path      : str = "outputs/audit_trail/drift_reports.jsonl",
    ) -> None:
        """Sauvegarde le rapport en mode append (JSONL — un rapport par ligne)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
