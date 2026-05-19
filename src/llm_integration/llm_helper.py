# -*- coding: utf-8 -*-
"""
src/llm_integration/llm_helper.py
===================================
Client LLM via API cloud gratuite pour la génération d'explications de fraudes.

Supporte (par ordre de priorité) :
  1. Groq       — llama3-8b-instruct  (rapide, gratuit, 14400 req/day)
  2. Gemini     — gemini-1.5-flash    (gratuit, 1500 req/day)
  3. HuggingFace — mistral-7b         (gratuit, rate limits variables)

Configuration : config/llm_config.yaml

Usage :
    from src.llm_integration.llm_helper import LLMHelper
    helper = LLMHelper()
    explanation = helper.explain_fraud(transaction, feature_errors, ae_score, threshold)
"""

import json
import yaml
from dotenv import load_dotenv
import os
import time
import requests
import numpy as np
from pathlib import Path
from typing import Optional
try:
    from json_repair import repair_json
except Exception:
    # json_repair is an optional helper to fix malformed JSON from LLMs.
    # Provide a permissive fallback that returns the original text so imports don't fail.
    def repair_json(text: str) -> str:
        return text
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()


def load_llm_config(path: str = "config/llm_config.yaml") -> dict:
    """Load the cloud LLM configuration and resolve the active API key."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    provider = config.get("active_provider", config.get("provider", "groq"))
    api_keys = config.get("api_keys") or {}
    raw_api_key = api_keys.get(provider, "")

    if isinstance(raw_api_key, str) and raw_api_key.startswith("${") and raw_api_key.endswith("}"):
        raw_api_key = os.getenv(raw_api_key[2:-1], raw_api_key)

    if provider == "groq":
        env_var = "GROQ_API_KEY"
    elif provider == "gemini":
        env_var = "GEMINI_API_KEY"
    else:
        env_var = "HF_API_KEY"

    api_key = os.getenv(env_var, raw_api_key)

    return {
        "provider": provider,
        "api_key": api_key,
        "generation": config.get("generation", {}),
        "pipeline": config.get("pipeline", {}),
        "models": config.get("models", {}),
    }

# ── Traduction features -> langage naturel ────────────────────────────────────
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

# ── Providers supportes ───────────────────────────────────────────────────────
PROVIDERS = {
    "groq": {
        "url"    : "https://api.groq.com/openai/v1/chat/completions",
        "model"  : "llama3-8b-8192",
        "header" : "Authorization",
        "prefix" : "Bearer ",
        "format" : "openai",
        "free"   : True,
        "limits" : "14 400 req/day, 6 000 tokens/min",
        "signup" : "https://console.groq.com",
    },
    "gemini": {
        "url"    : "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "model"  : "gemini-1.5-flash",
        "header" : None,          # cle dans query param
        "format" : "gemini",
        "free"   : True,
        "limits" : "1 500 req/day, 1M tokens/min",
        "signup" : "https://makersuite.google.com/app/apikey",
    },
    "huggingface": {
        "url"    : "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        "model"  : "mistralai/Mistral-7B-Instruct-v0.2",
        "header" : "Authorization",
        "prefix" : "Bearer ",
        "format" : "hf",
        "free"   : True,
        "limits" : "Rate limits variables selon le modele",
        "signup" : "https://huggingface.co/settings/tokens",
    },
}


class LLMHelper:
    """
    Client LLM cloud (Groq / Gemini / HuggingFace) pour expliquer les fraudes.

    Remplace OllamaHelper — aucune ressource locale requise.

    Parameters
    ----------
    provider : str
        'groq' (recommande) | 'gemini' | 'huggingface'
    api_key : str
        Cle API du provider choisi.
    timeout : int
        Timeout en secondes (defaut 30 — APIs cloud = rapides).
    temperature : float
        0.0 = deterministe | 1.0 = creatif.
    max_tokens : int
        Limite de tokens en sortie.
    feature_names : list
        Noms des 14 features dans l'ordre du modele.
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str = "",
        timeout: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 400,
        feature_names: Optional[list] = None,
        config_path: str = "config/llm_config.yaml",
    ):
        cfg = load_llm_config(config_path)

        provider = provider or cfg["provider"]
        api_key = api_key or cfg["api_key"]
        generation = cfg.get("generation", {})
        self.provider = provider
        self.api_key = api_key
        self.config = PROVIDERS[provider]

        self.timeout = timeout if timeout is not None else generation.get("timeout", 30)
        self.temperature = temperature if temperature is not None else generation.get("temperature", 0.1)
        self.max_tokens = max_tokens if max_tokens is not None else generation.get("max_tokens", 400)
        self.feature_names = feature_names or list(FEATURE_LABELS.keys())
        self._available = None

    # ── Disponibilite ─────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        """Verifie que la cle API est renseignee et le provider accessible."""
        if not self.api_key or self.api_key in ("", "YOUR_API_KEY", "VOTRE_CLE_ICI"):
            self._available = False
            return False
        if self._available is not None:
            return self._available
        try:
            # Test leger
            if self.provider == "groq":
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                self._available = r.status_code == 200
            elif self.provider == "gemini":
                r = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}",
                    timeout=5
                )
                self._available = r.status_code == 200
            elif self.provider == "huggingface":
                r = requests.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available


    def load_llm_config(path="config/llm_config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        provider = config["active_provider"]

        api_key = os.getenv(
            {
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "huggingface": "HF_API_KEY",
            }[provider]
        )

        return {
            "provider": provider,
            "api_key": api_key,
            "generation": config["generation"],
            "pipeline": config["pipeline"],
        }
        
    
    
    
    # ── Construction prompt ───────────────────────────────────────────────────
    def _build_prompt(
        self,
        transaction : dict,
        top_features: list,
        ae_score    : float,
        threshold   : float,
    ) -> str:
        tx_type = (
            "TRANSFER" if transaction.get("type_TRANSFER", 0) > 0.5 else
            "CASH_OUT" if transaction.get("type_CASH_OUT", 0) > 0.5 else
            "CASH_IN"  if transaction.get("type_CASH_IN", 0)  > 0.5 else
            "PAYMENT"  if transaction.get("type_PAYMENT", 0)  > 0.5 else "DEBIT"
        )
        log_amt    = transaction.get("log_amount", 0)
        amount_est = round(float(np.expm1(log_amt)), 2) if log_amt > 0 else 0
        hour       = int(round(transaction.get("hour", 0)))

        anomalies = ""
        for i, (feat, err, val) in enumerate(top_features, 1):
            label  = FEATURE_LABELS.get(feat, feat)
            interp = FEATURE_INTERPRETATIONS.get(feat, f"Valeur anormale ({err:.4f}).")
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

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3)
    )

    # ── Appel API ─────────────────────────────────────────────────────────────
    def _call_api(self, prompt: str) -> tuple[str, float]:
        """Envoie le prompt au provider selectionne. Retourne (texte, duree)."""
        t0 = time.time()

        if self.provider == "groq":
            resp = requests.post(
                self.config["url"],
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type" : "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": """
                            Tu es un analyste fraude senior spécialisé en audit bancaire.

                            Règles STRICTES :
                            - Explique uniquement les anomalies fournies.
                            - N'invente jamais d'informations absentes.
                            - Reste prudent et factuel.
                            - Ne parle jamais de certitude absolue.
                            - Produis uniquement un JSON valide.
                            - N'utilise jamais markdown.
                            - Base ton raisonnement uniquement sur les données fournies.
                            """
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                    "temperature": self.temperature,
                    "max_tokens" : self.max_tokens,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

        elif self.provider == "gemini":
            resp = requests.post(
                f"{self.config['url']}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {
                          "temperature"    : self.temperature,
                          "maxOutputTokens": self.max_tokens,
                      }},
                timeout=self.timeout,
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
                        "max_new_tokens": self.max_tokens,
                        "temperature"   : self.temperature,
                        "return_full_text": False,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            text = (result[0]["generated_text"] if isinstance(result, list)
                    else result.get("generated_text", str(result))).strip()
        else:
            raise ValueError(f"Provider inconnu : {self.provider}")

        return text, time.time() - t0

    # ── Parse JSON ────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        for marker in ("```json", "```"):
            if marker in text:
                text = text.split(marker)[1].split("```")[0]
                break
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        repaired = repair_json(text)
        return json.loads(repaired)

    # ── Interface principale ──────────────────────────────────────────────────
    def explain_fraud(
        self,
        transaction   : dict,
        feature_errors: np.ndarray,
        ae_score      : float,
        threshold     : float,
        top_k         : int = 3,
    ) -> dict:
        """Genere une explication LLM pour une transaction frauduleuse."""
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
            print("\n================ LLM PROMPT ================\n")
            print(prompt)

            print("\n================ RAW RESPONSE ================\n")
            print(raw)
            parsed = self._parse_json(raw)
            parsed.update({
                "ae_score"   : round(ae_score, 6),
                "threshold"  : round(threshold, 6),
                "top_features": [{"feature": n, "error": round(e, 6), "value": round(v, 6)}
                                  for n, e, v in top_features],
                "duration_s" : round(duration, 2),
                "status"     : "ok",
                "provider"   : self.provider,
                "model"      : self.config["model"],
            })
            return parsed
        except requests.Timeout:
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb.update({"status": "timeout", "duration_s": self.timeout})
            return fb
        except json.JSONDecodeError:
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb.update({"status": "json_parse_error"})
            return fb
        except requests.HTTPError as e:
            fb = self._fallback(
                transaction,
                top_features,
                ae_score,
                threshold
            )
            status_code = "?"
            error_detail = ""
            if e.response is not None:
                try:
                    status_code = e.response.status_code
                except:
                    pass
                try:
                    error_detail = e.response.text[:500]
                except:
                    pass
            print(f"\n[HTTP ERROR {status_code}]")
            print(error_detail)

            fb.update({
                "status": f"http_error_{status_code}",
                "error_detail": error_detail
            })
            return fb
        except Exception as e:
            fb = self._fallback(transaction, top_features, ae_score, threshold)
            fb.update({"status": f"error_{type(e).__name__}"})
            return fb

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _fallback(
        self,
        transaction : dict,
        top_features: list,
        ae_score    : float,
        threshold   : float,
    ) -> dict:
        risk = ("CRITIQUE" if ae_score > threshold * 3 else
                "ELEVE"    if ae_score > threshold * 1.5 else "MODERE")
        raisons = [
            FEATURE_INTERPRETATIONS.get(n, f"Anomalie sur {n} (erreur={e:.4f}).")
            for n, e, _ in top_features
        ]
        return {
            "risk_level"          : risk,
            "ae_score"            : round(ae_score, 6),
            "threshold"           : round(threshold, 6),
            "resume"              : (f"Transaction suspecte (score={ae_score:.2f}, seuil={threshold:.2f}). "
                                     f"Anomalies: {', '.join(n for n, _, _ in top_features)}."),
            "raisons"             : raisons,
            "actions_recommandees": [
                "Verifier manuellement la transaction.",
                "Contacter le titulaire du compte.",
                "Bloquer le compte si recidive detectee.",
            ],
            "confiance"           : "MOYENNE",
            "top_features"        : [{"feature": n, "error": round(e, 6), "value": round(v, 6)}
                                      for n, e, v in top_features],
            "status"              : "fallback_rule_based",
            "provider"            : self.provider,
            "model"               : "rule_based",
            "duration_s"          : 0.0,
        }

    # ── Batch ─────────────────────────────────────────────────────────────────
    def explain_batch(
        self,
        transactions        : list,
        feature_errors_batch: np.ndarray,
        ae_scores           : np.ndarray,
        threshold           : float,
        top_k               : int  = 3,
        verbose             : bool = True,
        max_explain         : int  = 20,
    ) -> list:
        """Explique un batch de transactions frauduleuses."""
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
                print(f"| {expl.get('risk_level','?'):8s} | "
                      f"{expl.get('status','?')} ({expl.get('duration_s',0):.1f}s)")

        if verbose:
            ok  = sum(1 for r in results if r.get("status") == "ok")
            fb  = sum(1 for r in results if "fallback" in str(r.get("status","")))
            err = len(results) - ok - fb
            print(f"\n  {len(results)} explications en {time.time()-t0:.1f}s")
            print(f"  ok={ok} | fallback={fb} | erreurs={err}")
        return results





