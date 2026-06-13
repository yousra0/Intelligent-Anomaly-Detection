"""Behavioral feature engineering helpers for client fraud detection."""

from __future__ import annotations

import pandas as pd


def add_balance_diff_orig(df: pd.DataFrame) -> pd.DataFrame:
	"""Add oldbalanceOrg - newbalanceOrig as a fraud-relevant behavior signal."""
	out = df.copy()
	out["balance_diff_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
	return out


def add_dest_zero_balance(df: pd.DataFrame) -> pd.DataFrame:
	"""Flag likely mule destinations with zero balance before and after transfer.

	nameDest check is skipped when the column is absent (e.g. anonymised CSVs),
	so the feature stays consistent between training and inference pipelines.
	"""
	out = df.copy()
	balances_zero = (out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)
	if "nameDest" in out.columns:
		dest_is_client = out["nameDest"].astype(str).str.startswith("C")
		out["dest_zero_balance"] = (dest_is_client & balances_zero).astype(int)
	else:
		out["dest_zero_balance"] = balances_zero.astype(int)
	return out

