"""Utilities for building Ollama prompts from fraud model outputs."""

from __future__ import annotations

import json


def build_fraud_explanation_prompt(
	transaction: dict,
	score: float,
	features: dict,
) -> str:
	"""Generate a structured prompt for LLM-based fraud explanation."""
	payload = {
		"transaction": {
			"amount": transaction.get("amount"),
			"type": transaction.get("type"),
			"hour": transaction.get("hour", features.get("hour")),
			"balance_diff_orig": features.get("balance_diff_orig"),
			"dest_zero_balance": features.get("dest_zero_balance"),
		},
		"model_score": float(score),
		"instructions": [
			"Explain why the transaction may be fraudulent.",
			"Reference each provided signal explicitly.",
			"Give a short risk level: low, medium, or high.",
			"Respond in French with concise bullet points.",
		],
	}

	return (
		"Tu es un analyste fraude. Analyse les signaux et justifie la decision.\n"
		"Reponds uniquement en JSON avec les cles: risk_level, reasons, actions.\n"
		f"Input:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
	)

