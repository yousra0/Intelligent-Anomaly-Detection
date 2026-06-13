# -*- coding: utf-8 -*-
"""
src/utils/model_governance.py
================================
Gouvernance et traçabilité des modèles de détection de fraude.

Fournit :
  - ModelRegistry : enregistre les versions de modèles (MRM)
  - AuditTrail    : journalise chaque décision de prédiction

Conformité visée : Basel III, FATF Recommendation 16, BCT circulaire 2021.

Usage :
    registry = ModelRegistry()
    registry.register(
        model_name="autoencoder_v1",
        metrics={"recall": 0.87, "f1": 0.83},
        artifacts={"weights": "outputs/models/autoencoder/autoencoder_weights.pt"},
    )

    trail = AuditTrail()
    trail.log_decision(
        transaction_id="TX_00042",
        model_name="autoencoder_v1",
        score=0.94,
        decision="FRAUD",
        threshold=0.48,
    )
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Chemins par défaut ────────────────────────────────────────────────────────

_AUDIT_DIR      = Path("outputs/audit_trail")
_REGISTRY_FILE  = _AUDIT_DIR / "model_registry.json"
_DECISIONS_FILE = _AUDIT_DIR / "decisions.jsonl"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: str) -> str:
    """Calcule le SHA-256 d'un fichier pour prouver qu'il n'a pas été altéré."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


# ── Registre de modèles (MRM) ─────────────────────────────────────────────────

class ModelRegistry:
    """
    Registre des versions de modèles (Model Risk Management).

    Chaque modèle enregistré a :
      - un identifiant unique (UUID)
      - ses métriques de validation (recall, F1, PR-AUC, coût TND)
      - les chemins vers ses artefacts avec leurs SHA-256
      - un statut de cycle de vie : active | deprecated | under_review
      - des dates de création, de validation, et de revalidation planifiée

    Le registre est stocké dans outputs/audit_trail/model_registry.json
    pour être consultable par les équipes d'audit indépendamment du code.
    """

    def __init__(self, registry_path: str | Path = _REGISTRY_FILE):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}, "active_model": None}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)

    def register(
        self,
        model_name       : str,
        metrics          : dict,
        artifacts        : dict[str, str],
        description      : str = "",
        validated_by     : str = "auto",
        revalidation_days: int = 180,
        cost_metrics     : Optional[dict] = None,
    ) -> str:
        """
        Enregistre un nouveau modèle dans le registre MRM.

        Parameters
        ----------
        model_name        : Nom unique du modèle (ex: "autoencoder_v1")
        metrics           : Dict de métriques {recall, f1, pr_auc, ...}
        artifacts         : Dict {label: chemin_fichier} des artefacts
        description       : Description textuelle du modèle
        validated_by      : Identifiant du validateur (personne ou processus)
        revalidation_days : Délai avant revalidation obligatoire (défaut 180j)
        cost_metrics      : Métriques coût métier {total_cost_tnd, fn_cost, ...}

        Returns
        -------
        model_id : UUID de l'entrée créée
        """
        model_id   = str(uuid.uuid4())
        now        = _now_utc()
        reval_date = datetime.now(timezone.utc)
        from datetime import timedelta
        reval_date = (reval_date + timedelta(days=revalidation_days)).isoformat()

        # Calcul des SHA-256 des artefacts
        artifact_checksums = {
            label: {"path": path, "sha256": _file_sha256(path)}
            for label, path in artifacts.items()
        }

        entry = {
            "model_id"              : model_id,
            "model_name"            : model_name,
            "description"           : description,
            "status"                : "active",
            "registered_at"         : now,
            "validated_by"          : validated_by,
            "revalidation_due"      : reval_date,
            "metrics"               : metrics,
            "cost_metrics"          : cost_metrics or {},
            "artifacts"             : artifact_checksums,
            "limitations"           : [
                "Entraîné sur données — à revalider sur données réelles.",
                "Performance dégradée si distribution des transactions change significativement (PSI > 0.25).",
                "Seuil de décision calibré sur données 2024 — à recalibrer semestriellement.",
            ],
            "revalidation_procedure": (
                "1. Collecter un mois de données récentes avec labels confirmés. "
                "2. Recalculer PSI vs données d'entraînement. "
                "3. Recalculer recall, F1 et coût TND sur nouvelles données. "
                "4. Si recall < 0.70 ou PSI max > 0.25 → réentraîner le modèle."
            ),
        }

        self._data["models"][model_id] = entry
        if self._data["active_model"] is None:
            self._data["active_model"] = model_id
        self._save()
        return model_id

    def deprecate(self, model_id: str, reason: str = "") -> None:
        """Marque un modèle comme déprécié."""
        if model_id in self._data["models"]:
            self._data["models"][model_id]["status"]       = "deprecated"
            self._data["models"][model_id]["deprecated_at"] = _now_utc()
            self._data["models"][model_id]["deprecation_reason"] = reason
            if self._data["active_model"] == model_id:
                self._data["active_model"] = None
            self._save()

    def get_active(self) -> Optional[dict]:
        """Retourne la fiche du modèle actif."""
        mid = self._data.get("active_model")
        return self._data["models"].get(mid) if mid else None

    def list_models(self) -> list[dict]:
        return list(self._data["models"].values())

    def print_registry(self) -> None:
        sep = "=" * 64
        print(sep)
        print("  REGISTRE DES MODÈLES — MODEL RISK MANAGEMENT")
        print(sep)
        for entry in self.list_models():
            icon = "✅" if entry["status"] == "active" else "🗑️"
            print(f"  {icon} [{entry['status'].upper():12s}] {entry['model_name']}")
            print(f"       ID           : {entry['model_id']}")
            print(f"       Enregistré   : {entry['registered_at'][:19]}")
            print(f"       Revalidation : {entry['revalidation_due'][:10]}")
            m = entry.get("metrics", {})
            print(f"       Recall={m.get('recall','?')}  F1={m.get('f1','?')}  PR-AUC={m.get('pr_auc','?')}")
            if entry.get("cost_metrics"):
                c = entry["cost_metrics"]
                print(f"       Coût total   : {c.get('total_cost_tnd', '?'):,.0f} TND")
        print(sep)


# ── Journal des décisions (Audit Trail) ───────────────────────────────────────

class AuditTrail:
    """
    Journalise chaque décision de prédiction du modèle en production.

    Chaque entrée contient :
      - identifiant unique de la décision (UUID)
      - transaction_id (référence métier)
      - modèle + version ayant pris la décision
      - score, seuil, décision (FRAUD / NORMAL)
      - timestamp UTC immuable
      - hash de la transaction (intégrité)

    Le fichier JSONL est append-only : on ne peut qu'ajouter,
    jamais modifier (garantie d'immuabilité pour l'audit).
    """

    def __init__(self, log_path: str | Path = _DECISIONS_FILE):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        transaction_id : str,
        model_name     : str,
        score          : float,
        decision       : str,
        threshold      : float,
        transaction_data: Optional[dict] = None,
        analyst_id     : str = "system",
        explanation_hash: Optional[str] = None,
    ) -> str:
        """
        Journalise une décision de prédiction.

        Parameters
        ----------
        transaction_id   : Identifiant de la transaction (ex: "TX_00042")
        model_name       : Modèle utilisé (ex: "autoencoder_v1")
        score            : Score d'anomalie brut
        decision         : "FRAUD" ou "NORMAL"
        threshold        : Seuil de décision utilisé
        transaction_data : Données de la transaction (pour hash d'intégrité)
        analyst_id       : Analyste ou système ayant déclenché la prédiction
        explanation_hash : Hash SHA-256 de l'explication LLM associée (si disponible)

        Returns
        -------
        decision_id : UUID de l'entrée journalisée
        """
        decision_id = str(uuid.uuid4())

        # Hash de la transaction pour prouver l'intégrité des données
        tx_hash = "unavailable"
        if transaction_data:
            tx_bytes = json.dumps(transaction_data, sort_keys=True).encode("utf-8")
            tx_hash  = hashlib.sha256(tx_bytes).hexdigest()

        record = {
            "decision_id"      : decision_id,
            "timestamp_utc"    : _now_utc(),
            "transaction_id"   : transaction_id,
            "analyst_id"       : analyst_id,
            "model_name"       : model_name,
            "score"            : round(float(score), 6),
            "threshold"        : round(float(threshold), 6),
            "decision"         : decision,
            "transaction_hash" : tx_hash,
            "explanation_hash" : explanation_hash,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        return decision_id

    def load_history(self, last_n: Optional[int] = None) -> list[dict]:
        """Charge l'historique des décisions depuis le fichier JSONL."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return records[-last_n:] if last_n else records

    def get_decisions_for_transaction(self, transaction_id: str) -> list[dict]:
        """Retourne toutes les décisions prises sur une transaction donnée."""
        return [r for r in self.load_history() if r["transaction_id"] == transaction_id]

    def print_recent(self, n: int = 10) -> None:
        records = self.load_history(last_n=n)
        sep = "=" * 72
        print(sep)
        print(f"  AUDIT TRAIL — {len(records)} dernières décisions")
        print(sep)
        print(f"  {'Timestamp':20s} {'TX ID':12s} {'Modèle':20s} {'Score':7s} {'Décision'}")
        print("-" * 72)
        for r in records:
            ts = r["timestamp_utc"][:19]
            icon = "🚨" if r["decision"] == "FRAUD" else "✅"
            print(f"  {ts:20s} {r['transaction_id']:12s} {r['model_name']:20s} "
                  f"{r['score']:7.4f} {icon} {r['decision']}")
        print(sep)
