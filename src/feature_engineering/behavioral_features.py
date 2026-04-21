"""Behavioral feature engineering helpers for PaySim fraud detection."""

from __future__ import annotations

import pandas as pd


def add_balance_diff_orig(df: pd.DataFrame) -> pd.DataFrame:
	"""Add oldbalanceOrg - newbalanceOrig as a fraud-relevant behavior signal."""
	out = df.copy()
	out["balance_diff_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
	return out


def add_dest_zero_balance(df: pd.DataFrame) -> pd.DataFrame:
	"""Flag likely mule destinations with zero balance before and after transfer."""
	out = df.copy()
	out["dest_zero_balance"] = (
		out["nameDest"].astype(str).str.startswith("C")
		& (out["oldbalanceDest"] == 0)
		& (out["newbalanceDest"] == 0)
	).astype(int)
	return out

