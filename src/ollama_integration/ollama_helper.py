"""
src/ollama_integration/ollama_helper.py
========================================
Client Ollama pour la génération d'explications de fraudes financières.

Usage :
    from src.ollama_integration.ollama_helper import OllamaHelper
    helper = OllamaHelper()
    explanation = helper.explain_fraud(transaction, feature_errors, ae_score, threshold)
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import Optional


# ── Traduction features → langage naturel ─────────────────────────────────────
FEATURE_LABELS = {
    "step"                  : "étape temporelle (heure absolue)",
    "hour"                  : "heure de la transaction",
    "day"                   : "jour de la transaction",
    "week"                  : "semaine de la transaction",
    "high_risk_hour"        : "heure à risque élevé (nuit/tôt matin)",
    "is_transfer_or_cashout": "type de transaction (TRANSFER ou CASH_OUT)",
    "balance_diff_orig"     : "différence de solde du compte émetteur",
    "dest_zero_balance"     : "compte destinataire avec solde initial nul",
    "type_CASH_IN"          : "type CASH_IN",
    "type_CASH_OUT"         : "type CASH_OUT",
    "type_DEBIT"            : "type DEBIT",
    "type_PAYMENT"          : "type PAYMENT",
    "type_TRANSFER"         : "type TRANSFER",
    "log_amount"            : "montant de la transaction (log-transformé)",
}

FEATURE_INTERPRETATIONS = {
    "balance_diff_orig"     : "Le solde du compte émetteur a été fortement diminué, signe potentiel de vidage de compte.",
    "high_risk_hour"        : "La transaction a eu lieu durant une plage horaire à haut risque (nuit ou tôt le matin).",
    "is_transfer_or_cashout": "La transaction est de type TRANSFER ou CASH_OUT, les seuls types associés à des fraudes.",
    "dest_zero_balance"     : "Le compte destinataire avait un solde initial de zéro, schéma typique d'un compte mule.",
    "log_amount"            : "Le montant de la transaction est anormalement élevé par rapport aux transactions normales.",
    "type_TRANSFER"         : "Les transferts directs sont le type de transaction le plus fréquemment impliqué dans les fraudes.",
    "type_CASH_OUT"         : "Les retraits en espèces constituent l'autre type de transaction impliqué dans les fraudes.",
    "hour"                  : "L'heure de la transaction correspond à une période d'activité inhabituelle.",
    "dest_zero_balance"     : "Le destinataire avait un solde nul, caractéristique d'un compte mule fraîchement créé.",
    "step"                  : "La transaction s'inscrit dans un schéma temporel anormal.",
    "day"                   : "Le jour de la transaction présente une anomalie temporelle.",
    "week"                  : "La semaine de la transaction présente une anomalie temporelle.",
    "type_CASH_IN"          : "Le type CASH_IN présente une reconstruction anormale.",
    "type_DEBIT"            : "Le type DEBIT présente une reconstruction anormale.",
    "type_PAYMENT"          : "Le type PAYMENT présente une reconstruction anormale.",
}


class OllamaHelper:
    """
    Client pour l'intégration Ollama dans le pipeline de détection de fraudes.

    Génère des explications en langage naturel pour les transactions
    détectées comme frauduleuses par l'AutoEncoder.

    Parameters
    ----------
    model : str
        Nom du modèle Ollama à utiliser (ex : 'llama3', 'mistral').
    base_url : str
        URL de base de l'API Ollama.
    timeout : int
        Timeout en secondes pour chaque requête.
    temperature : float
        Température du LLM (0.0 = déterministe, 1.0 = créatif).
    max_tokens : int
        Nombre maximum de tokens dans la réponse.
    feature_names : list
        Liste des noms de features dans l'ordre du modèle.
    """

    def __init__(
        self,
        model        : str   = "llama3",
        base_url     : str   = "http://localhost:11434",
        timeout      : int   = 60, #120
        temperature  : float = 0.2, #0.1
        max_tokens   : int   = 256, #512
        feature_names: Optional[list] = None,
    ):
        self.model         = model
        self.base_url      = base_url.rstrip("/")
        self.timeout       = timeout
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.feature_names = feature_names or list(FEATURE_LABELS.keys())
        self._available    = None  # cache du test de disponibilité

    # ── Disponibilité ──────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        """Vérifie si Ollama est démarré et accessible."""
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def list_models(self) -> list:
        """Retourne la liste des modèles disponibles dans Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Construction du prompt ─────────────────────────────────────────────────
    def _build_prompt(
        self,
        transaction    : dict,
        top_features   : list[tuple[str, float, float]],  # (name, error, value)
        ae_score       : float,
        threshold      : float,
        model_name     : str = "AutoEncoder",
    ) -> str:
        """
        Construit le prompt structuré pour l'explication de fraude.

        Parameters
        ----------
        transaction : dict
            Dictionnaire {feature_name: value} de la transaction originale.
        top_features : list of (name, error, value)
            Top-3 features avec la plus haute erreur de reconstruction.
        ae_score : float
            Score d'anomalie (erreur MSE) de la transaction.
        threshold : float
            Seuil de détection utilisé.
        model_name : str
            Nom du modèle utilisé pour la détection.
        """
        # Contexte transaction
        tx_type    = "TRANSFER"  if transaction.get("type_TRANSFER", 0)  > 0.5 else \
                     "CASH_OUT"  if transaction.get("type_CASH_OUT", 0)  > 0.5 else \
                     "CASH_IN"   if transaction.get("type_CASH_IN", 0)   > 0.5 else \
                     "PAYMENT"   if transaction.get("type_PAYMENT", 0)   > 0.5 else \
                     "DEBIT"
        log_amount = transaction.get("log_amount", 0)
        amount_est = round(np.expm1(log_amount), 2) if log_amount > 0 else 0
        hour       = int(round(transaction.get("hour", 0)))

        # Construction des anomalies détectées
        anomalies_txt = ""
        for i, (feat_name, feat_error, feat_value) in enumerate(top_features, 1):
            label  = FEATURE_LABELS.get(feat_name, feat_name)
            interp = FEATURE_INTERPRETATIONS.get(feat_name, "Valeur anormale par rapport aux transactions normales.")
            anomalies_txt += (
                f"  {i}. Feature : {label}\n"
                f"     Erreur de reconstruction : {feat_error:.4f}\n"
                f"     Interprétation : {interp}\n\n"
            )

        prompt = f"""Tu es un expert en détection de fraude financière. Analyse la transaction suivante qui a été détectée comme potentiellement frauduleuse par un système d'intelligence artificielle.

## Transaction analysée
- Type : {tx_type}
- Montant estimé : {amount_est:,.2f}
- Heure : {hour}h
- Score d'anomalie : {ae_score:.4f} (seuil : {threshold:.4f})

## Anomalies détectées par l'{model_name}
Les 3 caractéristiques les plus anormales de cette transaction (par rapport aux transactions légitimes connues) :

{anomalies_txt}
## Ta mission
En te basant exclusivement sur les anomalies détectées ci-dessus, génère une explication structurée en JSON avec exactement ce format :

{{
  "risk_level": "CRITIQUE" | "ÉLEVÉ" | "MODÉRÉ",
  "score_anomalie": {ae_score:.4f},
  "resume": "Une phrase résumant pourquoi cette transaction est suspecte.",
  "raisons": [
    "Raison 1 basée sur la première anomalie.",
    "Raison 2 basée sur la deuxième anomalie.",
    "Raison 3 basée sur la troisième anomalie."
  ],
  "actions_recommandees": [
    "Action concrète 1 pour l'analyste.",
    "Action concrète 2."
  ],
  "confiance": "HAUTE" | "MOYENNE" | "FAIBLE"
}}

Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans balises markdown."""

        return prompt

    # ── Appel API Ollama ───────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str) -> tuple[str, float]:
        """
        Envoie le prompt à Ollama et retourne (réponse_texte, durée_secondes).

        Returns
        -------
        (text, duration) : str, float
            Texte brut de la réponse et durée d'appel.
        """
        payload = {
            "model"  : self.model,
            "prompt" : prompt,
            "stream" : False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p"      : 0.9,
            },
        }
        t0 = time.time()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json    = payload,
            timeout = self.timeout,
        )
        duration = time.time() - t0
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text, duration

    # ── Parse JSON réponse ─────────────────────────────────────────────────────
    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """
        Parse la réponse JSON du LLM, robuste aux artefacts de formatage.

        Gère : backticks markdown, texte avant/après le JSON, JSON malformé.
        """
        # Supprimer les balises markdown
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Chercher le premier { ... } valide
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        return json.loads(text)

    # ── Interface principale ───────────────────────────────────────────────────
    def explain_fraud(
        self,
        transaction  : dict,
        feature_errors: np.ndarray,
        ae_score     : float,
        threshold    : float,
        top_k        : int = 3,
    ) -> dict:
        """
        Génère une explication complète pour une transaction frauduleuse.

        Parameters
        ----------
        transaction : dict
            {feature_name: valeur_scalée} — valeurs réelles de la transaction.
        feature_errors : np.ndarray, shape (14,)
            Erreur de reconstruction par feature (|valeur_réelle - valeur_reconstruite|).
        ae_score : float
            Score d'anomalie global (MSE) de la transaction.
        threshold : float
            Seuil de détection utilisé.
        top_k : int
            Nombre de features à inclure dans l'explication (défaut : 3).

        Returns
        -------
        dict avec les clés :
            - risk_level, resume, raisons, actions_recommandees, confiance
            - ae_score, threshold, top_features, duration_s, status
        """
        # ── Top-k features les plus anormales ─────────────────────────────────
        sorted_idx  = np.argsort(feature_errors)[::-1][:top_k]
        top_features = [
            (self.feature_names[i], float(feature_errors[i]), float(transaction.get(self.feature_names[i], 0)))
            for i in sorted_idx
        ]

        # ── Construction du prompt ─────────────────────────────────────────────
        prompt = self._build_prompt(transaction, top_features, ae_score, threshold)

        # ── Appel Ollama ───────────────────────────────────────────────────────
        if not self.is_available():
            return self._fallback_explanation(transaction, top_features, ae_score, threshold)

        try:
            raw_text, duration = self._call_ollama(prompt)
            parsed             = self._parse_json_response(raw_text)

            # Enrichissement avec les métadonnées
            parsed["ae_score"]     = round(ae_score, 6)
            parsed["threshold"]    = round(threshold, 6)
            parsed["top_features"] = [
                {"feature": n, "error": round(e, 6), "value": round(v, 6)}
                for n, e, v in top_features
            ]
            parsed["duration_s"]   = round(duration, 2)
            parsed["status"]       = "ok"
            parsed["model"]        = self.model

            return parsed

        except requests.Timeout:
            return {**self._fallback_explanation(transaction, top_features, ae_score, threshold),
                    "status": "timeout", "duration_s": self.timeout}
        except json.JSONDecodeError as e:
            return {**self._fallback_explanation(transaction, top_features, ae_score, threshold),
                    "status": "json_parse_error", "raw_response": raw_text[:500]}
        except Exception as e:
            return {**self._fallback_explanation(transaction, top_features, ae_score, threshold),
                    "status": f"error: {type(e).__name__}: {e}"}

    # ── Fallback sans LLM ──────────────────────────────────────────────────────
    def _fallback_explanation(
        self,
        transaction  : dict,
        top_features : list,
        ae_score     : float,
        threshold    : float,
    ) -> dict:
        """
        Génération d'explication règle-based quand Ollama n'est pas disponible.
        Permet de tester le pipeline sans LLM.
        """
        risk_level = "CRITIQUE" if ae_score > threshold * 3 else \
                     "ÉLEVÉ"    if ae_score > threshold * 1.5 else "MODÉRÉ"

        raisons = []
        for feat_name, feat_error, feat_value in top_features:
            interp = FEATURE_INTERPRETATIONS.get(
                feat_name,
                f"La feature '{feat_name}' présente une erreur de reconstruction élevée ({feat_error:.4f})."
            )
            raisons.append(interp)

        return {
            "risk_level"           : risk_level,
            "ae_score"             : round(ae_score, 6),
            "threshold"            : round(threshold, 6),
            "resume"               : f"Transaction suspecte avec score d'anomalie {ae_score:.4f} (seuil : {threshold:.4f}). "
                                     f"Anomalies détectées sur : {', '.join(f for f, _, _ in top_features)}.",
            "raisons"              : raisons,
            "actions_recommandees" : [
                "Vérifier manuellement les détails de la transaction.",
                "Contacter le titulaire du compte pour confirmation.",
                "Bloquer temporairement le compte si plusieurs transactions similaires sont détectées.",
            ],
            "confiance"            : "MOYENNE",
            "top_features"        : [
                {"feature": n, "error": round(e, 6), "value": round(v, 6)}
                for n, e, v in top_features
            ],
            "status"               : "fallback_no_llm",
            "model"                : "rule_based",
            "duration_s"           : 0.0,
        }

    # ── Batch processing ───────────────────────────────────────────────────────
    def explain_batch(
        self,
        transactions   : list[dict],
        feature_errors_batch: np.ndarray,
        ae_scores      : np.ndarray,
        threshold      : float,
        top_k          : int = 3,
        verbose        : bool = True,
        max_explain    : int = 50,
    ) -> list[dict]:
        """
        Génère des explications pour un batch de transactions frauduleuses.

        Parameters
        ----------
        transactions : list[dict]
            Liste de transactions détectées comme fraudes.
        feature_errors_batch : np.ndarray, shape (N, 14)
            Erreurs de reconstruction par feature pour chaque transaction.
        ae_scores : np.ndarray, shape (N,)
            Scores d'anomalie pour chaque transaction.
        threshold : float
            Seuil de détection.
        top_k : int
            Nombre de features à expliquer par transaction.
        verbose : bool
            Afficher la progression.
        max_explain : int
            Nombre maximum de transactions à expliquer (les plus anormales en premier).

        Returns
        -------
        list[dict] : explications pour chaque transaction
        """
        n = len(transactions)

        # Trier par score décroissant — expliquer les plus anormales en premier
        sorted_idx = np.argsort(ae_scores)[::-1][:max_explain]

        results = []
        t_total = time.time()

        for i, idx in enumerate(sorted_idx):
            if verbose:
                print(f"  [{i+1:3d}/{len(sorted_idx)}] Transaction idx={idx} "
                      f"| score={ae_scores[idx]:.4f} ", end="", flush=True)

            expl = self.explain_fraud(
                transaction    = transactions[idx],
                feature_errors = feature_errors_batch[idx],
                ae_score       = float(ae_scores[idx]),
                threshold      = threshold,
                top_k          = top_k,
            )
            expl["transaction_idx"] = int(idx)
            results.append(expl)

            if verbose:
                status = expl.get("status", "?")
                dur    = expl.get("duration_s", 0)
                risk   = expl.get("risk_level", "?")
                print(f"| {risk:8s} | {status} ({dur:.1f}s)")

        elapsed = time.time() - t_total
        if verbose:
            ok  = sum(1 for r in results if r.get("status") == "ok")
            fb  = sum(1 for r in results if "fallback" in r.get("status",""))
            err = len(results) - ok - fb
            print(f"\n  Total : {len(results)} explications en {elapsed:.1f}s")
            print(f"  ✅ LLM : {ok}  |  ⚠ Fallback : {fb}  |  ❌ Erreurs : {err}")

        return results



