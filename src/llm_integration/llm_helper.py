# -*- coding: utf-8 -*-
"""
src/llm_integration/llm_helper.py
===================================
Client LLM cloud pour la génération d'explications de fraudes financières.

Providers supportés (configurer dans config/llm_config.yaml) :
  - Groq        : llama-3.3-70b-versatile  — rapide, gratuit, 14 400 req/jour
  - Gemini      : gemini-1.5-flash         — gratuit, 1 500 req/jour
  - HuggingFace : Mistral-7B-Instruct      — gratuit, rate limits variables

Flux d'appel :
  LLMHelper.explain_fraud()
      └─> _build_prompt()     : construit le prompt structuré
      └─> _call_api()         : appel HTTP vers le provider (avec retry)
      └─> _parse_json()       : extraction et réparation du JSON retourné
      └─> _fallback()         : explication rule-based si le LLM est indisponible

Usage :
    from src.llm_integration.llm_helper import LLMHelper
    helper = LLMHelper()
    explanation = helper.explain_fraud(transaction, feature_errors, ae_score, threshold)
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception

try:
    from json_repair import repair_json
except ImportError:
    # json_repair est optionnel : corrige les JSON mal formés renvoyés par les LLMs.
    # Sans lui, on renvoie le texte brut et json.loads lèvera une JSONDecodeError
    # si la réponse est vraiment invalide.
    def repair_json(text: str) -> str:
        return text

# Charge les variables d'environnement depuis .env (clés API, etc.)
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

def load_llm_config(path: str = "config/llm_config.yaml") -> dict:
    """
    Charge la configuration LLM depuis le fichier YAML et résout la clé API active.

    Priorité pour la clé API (du plus au moins prioritaire) :
      1. Variable d'environnement (ex: GROQ_API_KEY)
      2. Valeur littérale dans api_keys.<provider> du YAML
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    provider = config.get("active_provider", config.get("provider", "groq"))

    # Résolution du placeholder ${VAR_NAME} éventuellement présent dans le YAML
    api_keys = config.get("api_keys") or {}
    raw_api_key = api_keys.get(provider, "")
    if isinstance(raw_api_key, str) and raw_api_key.startswith("${") and raw_api_key.endswith("}"):
        raw_api_key = os.getenv(raw_api_key[2:-1], raw_api_key)

    env_var_map = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "huggingface": "HF_API_KEY"}
    api_key = os.getenv(env_var_map.get(provider, ""), raw_api_key)

    return {
        "provider"   : provider,
        "api_key"    : api_key,
        "generation" : config.get("generation", {}),
        "pipeline"   : config.get("pipeline", {}),
        "models"     : config.get("models", {}),
        # ssl_verify=False nécessaire sur les réseaux d'entreprise avec proxy SSL
        "ssl_verify" : config.get("ssl_verify", True),
    }


# ── Référentiels des features ──────────────────────────────────────────────────

# Traduction technique -> langage naturel injecté dans le prompt LLM
FEATURE_LABELS = {
    "step"                  : "etape temporelle (heure absolue)",
    "hour"                  : "heure de la transaction",
    "day"                   : "jour de la transaction",
    "week"                  : "semaine de la transaction",
    "high_risk_hour"        : "heure a risque eleve (nuit/tot matin)",
    "is_transfer_or_cashout": "type de transaction (TRANSFER ou CASH_OUT)",
    "balance_diff_orig"     : "difference de solde du compte emetteur",
    "dest_zero_balance"     : "compte destinataire avec solde initial nul",
    "type_CASH_IN"          : "type CASH_IN",
    "type_CASH_OUT"         : "type CASH_OUT",
    "type_DEBIT"            : "type DEBIT",
    "type_PAYMENT"          : "type PAYMENT",
    "type_TRANSFER"         : "type TRANSFER",
    "log_amount"            : "montant de la transaction (log-transforme)",
}

# Interprétations métier utilisées dans le fallback rule-based et dans le prompt
FEATURE_INTERPRETATIONS = {
    "balance_diff_orig"     : "Le solde du compte emetteur a ete fortement diminue, signe potentiel de vidage de compte.",
    "high_risk_hour"        : "La transaction a eu lieu durant une plage horaire a haut risque (nuit ou tot le matin).",
    "is_transfer_or_cashout": "La transaction est de type TRANSFER ou CASH_OUT, les seuls types associes a des fraudes.",
    "dest_zero_balance"     : "Le compte destinataire avait un solde initial de zero, configuration parfois observee dans certains scenarios de fraude.",
    "log_amount"            : "Le montant de la transaction est anormalement eleve par rapport aux transactions normales.",
    "type_TRANSFER"         : "Dans le dataset d'entrainement, les transactions TRANSFER apparaissent frequemment parmi les anomalies detectees.",
    "type_CASH_OUT"         : "Les retraits en especes constituent l'autre type de transaction implique dans les fraudes.",
    "hour"                  : "L'heure de la transaction correspond a une periode d'activite inhabituelle.",
    "step"                  : "La transaction s'inscrit dans un schema temporel anormal.",
    "day"                   : "Le jour de la transaction presente une anomalie temporelle.",
    "week"                  : "La semaine de la transaction presente une anomalie temporelle.",
    "type_CASH_IN"          : "Le type CASH_IN presente une reconstruction anormale.",
    "type_DEBIT"            : "Le type DEBIT presente une reconstruction anormale.",
    "type_PAYMENT"          : "Le type PAYMENT presente une reconstruction anormale.",
}


# ── Registre des providers ─────────────────────────────────────────────────────

# Métadonnées statiques par provider : URL d'endpoint, format de réponse, limites gratuites.
# Le modèle effectivement utilisé est lu depuis llm_config.yaml (clé `models.<provider>`).
PROVIDERS = {
    "groq": {
        "url"    : "https://api.groq.com/openai/v1/chat/completions",
        "model"  : "llama-3.3-70b-versatile",   # fallback si absent du yaml
        "format" : "openai",
        "free"   : True,
        "limits" : "14 400 req/jour | 6 000 tokens/min",
        "signup" : "https://console.groq.com",
    },
    "gemini": {
        "url"    : "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model"  : "gemini-1.5-flash",
        "format" : "gemini",
        "free"   : True,
        "limits" : "1 500 req/jour | 1 000 000 tokens/min",
        "signup" : "https://makersuite.google.com/app/apikey",
    },
    "huggingface": {
        "url"    : "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        "model"  : "mistralai/Mistral-7B-Instruct-v0.2",
        "format" : "hf",
        "free"   : True,
        "limits" : "Rate limits variables selon le modele",
        "signup" : "https://huggingface.co/settings/tokens",
    },
}


def _is_retryable(exc: Exception) -> bool:
    """Retente uniquement sur 429 (rate limit) et 5xx (erreur serveur). Pas sur 400/401/403/404."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return not isinstance(exc, requests.HTTPError)


# ── Classe principale ──────────────────────────────────────────────────────────

class LLMHelper:
    """
    Client LLM cloud (Groq / Gemini / HuggingFace) pour expliquer les fraudes.

    Génère, pour chaque transaction suspecte, une explication JSON structurée
    comprenant : niveau de risque, résumé, raisons et actions recommandées.
    Si le LLM est indisponible, un fallback rule-based produit une explication
    déterministe à partir des interprétations métier pré-définies.

    Parameters
    ----------
    provider : str | None
        'groq' | 'gemini' | 'huggingface'. Si None, lu depuis llm_config.yaml.
    api_key : str
        Clé API. Si vide, résolue depuis llm_config.yaml / variables d'environnement.
    timeout : int
        Timeout HTTP en secondes.
    temperature : float
        0.0 = déterministe, 1.0 = créatif. Valeur basse recommandée pour JSON strict.
    max_tokens : int
        Limite de tokens en sortie (400 suffisent pour le JSON attendu).
    feature_names : list | None
        Noms des features dans l'ordre du modèle. Par défaut : FEATURE_LABELS.keys().
    config_path : str
        Chemin vers llm_config.yaml.
    """

    def __init__(
        self,
        provider     : str | None = None,
        api_key      : str = "",
        timeout      : int = 30,
        temperature  : float = 0.1,
        max_tokens   : int = 400,
        feature_names: Optional[list] = None,
        config_path  : str = "config/llm_config.yaml",
    ):
        cfg = load_llm_config(config_path)

        self.provider      = provider or cfg["provider"]
        # Rejette les placeholders non résolus (ex: "${GROQ_API_KEY}" si load_dotenv()
        # n'a pas été appelé avant os.getenv dans le notebook) et retombe sur cfg.
        _is_placeholder = isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}")
        self.api_key       = (api_key if api_key and not _is_placeholder else None) or cfg["api_key"]
        self.config        = PROVIDERS[self.provider]
        self.timeout       = timeout
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.feature_names = feature_names or list(FEATURE_LABELS.keys())
        self.ssl_verify    = cfg.get("ssl_verify", True)
        # Modèle lu depuis le yaml ; fallback sur la valeur du registre PROVIDERS
        self.model         = cfg.get("models", {}).get(self.provider) or self.config["model"]
        # Cache du résultat du test de disponibilité (évite un appel réseau par transaction)
        self._available    = None

    # ── Disponibilité ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Vérifie que la clé API est renseignée et que le provider répond.

        Le résultat est mis en cache après le premier appel pour éviter
        de solliciter l'API à chaque transaction du batch.
        """
        if not self.api_key or self.api_key in ("", "YOUR_API_KEY", "VOTRE_CLE_ICI") or \
                (self.api_key.startswith("${") and self.api_key.endswith("}")):
            self._available = False
            return False

        if self._available is not None:
            return self._available

        # Requête légère (liste des modèles / whoami) pour valider la clé
        try:
            if self.provider == "groq":
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                    verify=self.ssl_verify,
                )
            elif self.provider == "gemini":
                r = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}",
                    timeout=5,
                    verify=self.ssl_verify,
                )
            elif self.provider == "huggingface":
                r = requests.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                    verify=self.ssl_verify,
                )
            else:
                self._available = False
                return False

            self._available = r.status_code == 200
        except Exception:
            self._available = False

        return self._available

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(
        self,
        transaction : dict,
        top_features: list,
        ae_score    : float,
        threshold   : float,
    ) -> str:
        """
        Construit le prompt envoyé au LLM.

        Reconstitue le type de transaction et le montant réel (inverse du log)
        pour que le LLM travaille sur des valeurs compréhensibles, pas sur
        les valeurs normalisées du modèle.
        """
        # Déduction du type de transaction depuis les colonnes one-hot
        tx_type = (
            "TRANSFER" if transaction.get("type_TRANSFER", 0) > 0.5 else
            "CASH_OUT" if transaction.get("type_CASH_OUT", 0)  > 0.5 else
            "CASH_IN"  if transaction.get("type_CASH_IN", 0)   > 0.5 else
            "PAYMENT"  if transaction.get("type_PAYMENT", 0)   > 0.5 else
            "DEBIT"
        )

        # Montant réel estimé (log_amount = log1p(montant), d'où expm1 pour inverser)
        log_amt    = transaction.get("log_amount", 0)
        amount_est = round(float(np.expm1(log_amt)), 2) if log_amt > 0 else 0
        hour       = int(round(transaction.get("hour", 0)))

        # Construction de la liste des anomalies avec label et interprétation métier
        anomalies = ""
        for i, (feat, err, _val) in enumerate(top_features, 1):
            label  = FEATURE_LABELS.get(feat, feat)
            interp = FEATURE_INTERPRETATIONS.get(feat, f"Valeur anormale (erreur={err:.4f}).")
            anomalies += f"  {i}. {label} (erreur={err:.4f}) : {interp}\n"

        return f"""Tu es un expert en fraude financiere pour PwC Tunisie. Analyse cette transaction suspecte et reponds UNIQUEMENT en JSON valide. IMPORTANT :
                - N'invente aucune information.
                - Utilise uniquement les anomalies fournies.
                - Ne suppose jamais une fraude certaine.
                - Les anomalies indiquent un comportement atypique, pas une preuve definitive de fraude.

Transaction : type={tx_type} | montant={amount_est:,.0f} TND | heure={hour}h
Score anomalie : {ae_score:.4f} (seuil={threshold:.4f})

Anomalies detectees :
{anomalies}
Reponds avec ce JSON exact (sans markdown, sans texte avant/apres) :
{{
  "risk_level": "CRITIQUE",
  "risk_score": 0.95,
  "confidence_score": 0.88,
  "resume": "Une phrase concise.",
  "raisons": [
    "raison 1",
    "raison 2",
    "raison 3"
  ],
  "actions_recommandees": [
    "action 1",
    "action 2"
  ]
}}"""

    # ── Appel API (avec retry automatique) ────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),  # 5s → 10s → 20s, adapté au 429
        retry=retry_if_exception(_is_retryable),
    )
    def _call_api(self, prompt: str) -> tuple[str, float]:
        """
        Envoie le prompt au provider sélectionné et retourne (texte_brut, durée_s).

        Lève une exception en cas d'erreur HTTP ou réseau ; tenacity réessaie
        automatiquement avant de lever RetryError, capturé dans explain_fraud.
        """
        t0 = time.time()

        if self.provider == "groq":
            resp = requests.post(
                self.config["url"],
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type" : "application/json",
                },
                json={
                    "model"      : self.model,
                    "messages"   : [
                        {
                            "role"   : "system",
                            "content": (
                                "Tu es un analyste fraude senior spécialisé en audit bancaire. "
                                "Règles STRICTES : explique uniquement les anomalies fournies, "
                                "n'invente jamais d'informations absentes, reste prudent et factuel, "
                                "ne parle jamais de certitude absolue, produis uniquement un JSON valide, "
                                "n'utilise jamais markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens" : self.max_tokens,
                },
                timeout=self.timeout,
                verify=self.ssl_verify,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

        elif self.provider == "gemini":
            resp = requests.post(
                f"{self.config['url']}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents"       : [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature"    : self.temperature,
                        "maxOutputTokens": self.max_tokens,
                    },
                },
                timeout=self.timeout,
                verify=self.ssl_verify,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif self.provider == "huggingface":
            resp = requests.post(
                self.config["url"],
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs"    : prompt,
                    "parameters": {
                        "max_new_tokens"  : self.max_tokens,
                        "temperature"     : self.temperature,
                        "return_full_text": False,
                    },
                },
                timeout=self.timeout,
                verify=self.ssl_verify,
            )
            resp.raise_for_status()
            result = resp.json()
            text = (
                result[0]["generated_text"] if isinstance(result, list)
                else result.get("generated_text", str(result))
            ).strip()

        else:
            raise ValueError(f"Provider inconnu : {self.provider}")

        return text, time.time() - t0

    # ── Parsing de la réponse JSON ────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Extrait et parse le JSON de la réponse brute du LLM.

        Les LLMs enveloppent parfois leur réponse dans des balises markdown (```json).
        On retire ces balises, isole le bloc JSON, puis on tente une réparation
        avec json_repair avant de parser.
        """
        text = text.strip()

        # Suppression des balises markdown éventuelles
        for marker in ("```json", "```"):
            if marker in text:
                text = text.split(marker)[1].split("```")[0]
                break

        # Isolation du premier bloc JSON valide
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        return json.loads(repair_json(text))

    # ── Interface principale ──────────────────────────────────────────────────

    def explain_fraud(
        self,
        transaction   : dict,
        feature_errors: np.ndarray,
        ae_score      : float,
        threshold     : float,
        top_k         : int = 3,
    ) -> dict:
        """
        Génère une explication pour une transaction détectée comme frauduleuse.

        Sélectionne les top_k features avec la plus grande erreur de reconstruction,
        construit le prompt, appelle le LLM et enrichit la réponse avec les
        métadonnées du modèle. En cas d'échec, retourne une explication fallback.

        Returns
        -------
        dict avec les clés : risk_level, resume, raisons, actions_recommandees,
             top_features, ae_score, threshold, status, provider, model, duration_s.
        """
        # Sélection des features les plus anormales (erreur de reconstruction élevée)
        sorted_idx   = np.argsort(feature_errors)[::-1][:top_k]
        top_features = [
            (self.feature_names[i], float(feature_errors[i]),
             float(transaction.get(self.feature_names[i], 0)))
            for i in sorted_idx
        ]
        prompt = self._build_prompt(transaction, top_features, ae_score, threshold)

        if not self.is_available():
            return self._fallback(transaction, top_features, ae_score, threshold)

        try:
            raw, duration = self._call_api(prompt)
            parsed = self._parse_json(raw)
            parsed.update({
                "ae_score"    : round(ae_score, 6),
                "threshold"   : round(threshold, 6),
                "top_features": [
                    {"feature": n, "error": round(e, 6), "value": round(v, 6)}
                    for n, e, v in top_features
                ],
                "duration_s"  : round(duration, 2),
                "status"      : "ok",
                "provider"    : self.provider,
                "model"       : self.model,
            })
            # Tampon d'audit : hash + timestamp pour traçabilité en cas de litige
            self._audit_stamp(prompt, raw, parsed)
            self._save_audit_log(parsed)
            return parsed

        except RetryError as e:
            # Toutes les tentatives ont échoué : on remonte la cause réelle
            cause = e.last_attempt.exception()
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb["status"] = f"retry_exhausted_{type(cause).__name__}"
            return fb

        except requests.Timeout:
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb.update({"status": "timeout", "duration_s": float(self.timeout)})
            return fb

        except json.JSONDecodeError:
            # Le LLM a répondu mais pas en JSON valide (même après réparation)
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb["status"] = "json_parse_error"
            return fb

        except requests.HTTPError as e:
            # Erreur HTTP (4xx/5xx) : on logue le détail pour diagnostiquer
            status_code  = e.response.status_code if e.response is not None else "?"
            error_detail = (e.response.text[:500] if e.response is not None else "")
            print(f"\n[HTTP ERROR {status_code}] {error_detail}")
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb.update({"status": f"http_error_{status_code}", "error_detail": error_detail})
            return fb

        except Exception as e:
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb["status"] = f"error_{type(e).__name__}"
            return fb

    # ── Fallback rule-based ───────────────────────────────────────────────────

    def _fallback(
        self,
        transaction : dict,
        top_features: list,
        ae_score    : float,
        threshold   : float,
    ) -> dict:
        """
        Génère une explication déterministe sans LLM.

        Utilisé quand le LLM est indisponible ou a échoué. Le niveau de risque
        est calculé par rapport au seuil AE : score > 3×seuil → CRITIQUE,
        score > 1.5×seuil → ELEVE, sinon MODERE.
        """
        risk = (
            "CRITIQUE" if ae_score > threshold * 3   else
            "ELEVE"    if ae_score > threshold * 1.5 else
            "MODERE"
        )
        raisons = [
            FEATURE_INTERPRETATIONS.get(n, f"Anomalie sur {n} (erreur={e:.4f}).")
            for n, e, _ in top_features
        ]
        return {
            "risk_level"          : risk,
            "ae_score"            : round(ae_score, 6),
            "threshold"           : round(threshold, 6),
            "resume"              : (
                f"Transaction suspecte (score={ae_score:.2f}, seuil={threshold:.2f}). "
                f"Anomalies: {', '.join(n for n, _, _ in top_features)}."
            ),
            "raisons"             : raisons,
            "actions_recommandees": [
                "Verifier manuellement la transaction.",
                "Contacter le titulaire du compte.",
                "Bloquer le compte si recidive detectee.",
            ],
            "confiance"           : "MOYENNE",
            "top_features"        : [
                {"feature": n, "error": round(e, 6), "value": round(v, 6)}
                for n, e, v in top_features
            ],
            "status"              : "fallback_rule_based",
            "provider"            : self.provider,
            "model"               : "rule_based",
            "duration_s"          : 0.0,
        }

    # ── Auditabilité ─────────────────────────────────────────────────────────

    def _audit_stamp(self, prompt: str, response_text: str, explanation: dict) -> dict:
        """
        Ajoute un tampon d'audit à une explication LLM :
          - hash SHA-256 du couple (prompt, réponse) — garantit la reproductibilité
          - timestamp UTC ISO-8601
          - version du modèle LLM utilisé

        Ces champs permettent de prouver, lors d'un litige, que l'explication
        n'a pas été modifiée après coup.
        """
        content    = (prompt + response_text).encode("utf-8")
        audit_hash = hashlib.sha256(content).hexdigest()
        explanation["_audit"] = {
            "hash"          : audit_hash,
            "timestamp_utc" : datetime.now(timezone.utc).isoformat(),
            "provider"      : self.provider,
            "model_version" : self.model,
            "hash_algo"     : "sha256",
        }
        return explanation

    def _save_audit_log(
        self,
        explanation: dict,
        path       : str = "outputs/audit_trail/explanations_audit.jsonl",
    ) -> None:
        """
        Persiste l'explication (avec son tampon d'audit) dans un fichier JSONL
        append-only. Chaque ligne = une explication immuable et traçable.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(explanation, ensure_ascii=False, default=str) + "\n")

    # ── Traitement batch ──────────────────────────────────────────────────────

    def explain_batch(
        self,
        transactions        : list,
        feature_errors_batch: np.ndarray,
        ae_scores           : np.ndarray,
        threshold           : float,
        top_k               : int   = 3,
        verbose             : bool  = True,
        max_explain         : int   = 20,
        delay_s             : float = 2.0,
    ) -> list:
        """
        Génère des explications pour un ensemble de transactions suspectes.

        Les transactions sont triées par score décroissant (les plus anormales
        en premier) et limitées à max_explain pour contrôler les coûts d'API.

        Returns
        -------
        list[dict] : une explication par transaction, chacune enrichie de
                     transaction_idx (position dans le batch d'entrée).
        """
        # Traitement par ordre décroissant de score d'anomalie
        sorted_idx = np.argsort(ae_scores)[::-1][:max_explain]
        results    = []
        t0         = time.time()

        for i, idx in enumerate(sorted_idx):
            if verbose:
                print(f"  [{i+1:3d}/{len(sorted_idx)}] idx={idx} score={ae_scores[idx]:.2f} ",
                      end="", flush=True)

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
                print(f"| {expl.get('risk_level', '?'):8s} | "
                      f"{expl.get('status', '?')} ({expl.get('duration_s', 0):.1f}s)")

            # Délai entre requêtes pour respecter le rate limit Groq (30 req/min free tier)
            if delay_s > 0 and i < len(sorted_idx) - 1:
                time.sleep(delay_s)

        if verbose:
            ok  = sum(1 for r in results if r.get("status") == "ok")
            fb  = sum(1 for r in results if "fallback" in str(r.get("status", "")))
            err = len(results) - ok - fb
            print(f"\n  {len(results)} explications en {time.time() - t0:.1f}s")
            print(f"  ok={ok} | fallback={fb} | erreurs={err}")

        return results
