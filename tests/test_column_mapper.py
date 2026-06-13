"""
tests/test_column_mapper.py
Tests du moteur de détection sémantique des colonnes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.column_mapper import map_columns, ColumnMapper, _score_column, SEMANTIC_FIELDS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Crée un DataFrame minimal avec les colonnes fournies."""
    n = 3
    base = {col: [val] * n for col, val in kwargs.items()}
    return pd.DataFrame(base)


def _paysim_renamed(**rename) -> pd.DataFrame:
    """
    Crée un DataFrame PaySim avec des colonnes renommées selon `rename`.
    rename = {nouveau_nom: nom_canonique}
    """
    canonical = {
        "step": 1, "type": "TRANSFER", "amount": 10000.0,
        "nameOrig": "C001", "oldbalanceOrg": 10000.0, "newbalanceOrig": 0.0,
        "nameDest": "M001", "oldbalanceDest": 0.0, "newbalanceDest": 10000.0,
    }
    data = {}
    # Appliquer le rename inverse
    inverse_rename = {v: k for k, v in rename.items()}
    for canonical_col, val in canonical.items():
        new_col = inverse_rename.get(canonical_col, canonical_col)
        data[new_col] = [val] * 3
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de scoring individuel
# ─────────────────────────────────────────────────────────────────────────────

class TestScoring:

    def test_exact_alias_score_is_max(self):
        assert _score_column("amount", SEMANTIC_FIELDS["amount"]) == 1.0
        assert _score_column("step", SEMANTIC_FIELDS["step"]) == 1.0
        assert _score_column("type", SEMANTIC_FIELDS["type"]) == 1.0

    def test_normalized_alias(self):
        score = _score_column("transaction_amount", SEMANTIC_FIELDS["amount"])
        assert score >= 0.90

    def test_french_alias_amount(self):
        score = _score_column("montant", SEMANTIC_FIELDS["amount"])
        assert score >= 0.90

    def test_abbreviated_amount(self):
        score = _score_column("amt", SEMANTIC_FIELDS["amount"])
        assert score >= 0.90

    def test_forbidden_token_eliminates(self):
        # "oldbalance" contient "bal" qui est interdit pour "amount"
        score = _score_column("oldbalance", SEMANTIC_FIELDS["amount"])
        assert score == 0.0

    def test_balance_discrimination_old_new(self):
        """oldbalanceOrg doit scorer plus haut que newbalanceOrig sur oldbalanceOrg."""
        score_old = _score_column("oldbalanceOrg", SEMANTIC_FIELDS["oldbalanceOrg"])
        score_new = _score_column("oldbalanceOrg", SEMANTIC_FIELDS["newbalanceOrig"])
        assert score_old > score_new

    def test_balance_discrimination_orig_dest(self):
        """oldbalanceOrg doit scorer plus haut que oldbalanceDest sur oldbalanceOrg."""
        score_orig = _score_column("oldbalanceOrg", SEMANTIC_FIELDS["oldbalanceOrg"])
        score_dest = _score_column("oldbalanceOrg", SEMANTIC_FIELDS["oldbalanceDest"])
        assert score_orig > score_dest


# ─────────────────────────────────────────────────────────────────────────────
# Tests de mapping complet
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnMapper:

    # ── Noms PaySim canoniques ────────────────────────────────────────────

    def test_canonical_paysim_names(self):
        df = _paysim_renamed()  # aucun rename
        result = map_columns(df)
        assert result.success
        for canonical in ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                          "oldbalanceDest", "newbalanceDest"]:
            assert canonical in result.df.columns

    # ── Noms anglais alternatifs ──────────────────────────────────────────

    def test_amount_variants(self):
        for col in ["montant", "amt", "transaction_amount", "value", "transaction_value"]:
            df = _paysim_renamed(**{col: "amount"})
            r = map_columns(df)
            assert "amount" in r.mapping, f"'{col}' non reconnu comme amount"
            assert r.mapping["amount"] == col

    def test_step_variants(self):
        for col in ["timestamp", "time_step", "period", "date"]:
            df = _paysim_renamed(**{col: "step"})
            r = map_columns(df)
            assert "step" in r.mapping, f"'{col}' non reconnu comme step"

    def test_type_variants(self):
        for col in ["transaction_type", "category", "kind", "nature"]:
            df = _paysim_renamed(**{col: "type"})
            r = map_columns(df)
            assert "type" in r.mapping, f"'{col}' non reconnu comme type"

    # ── Noms français ─────────────────────────────────────────────────────

    def test_french_columns(self):
        df = _paysim_renamed(
            montant="amount",
            type_transaction="type",
            solde_avant="oldbalanceOrg",
            solde_apres="newbalanceOrig",
        )
        r = map_columns(df)
        assert r.mapping.get("amount") == "montant"
        assert r.mapping.get("type") == "type_transaction"

    # ── Colonnes de balance ────────────────────────────────────────────────

    def test_balance_columns_french(self):
        df = _paysim_renamed(
            solde_avant_source="oldbalanceOrg",
            solde_apres_source="newbalanceOrig",
            solde_avant_dest="oldbalanceDest",
            solde_apres_dest="newbalanceDest",
        )
        r = map_columns(df)
        assert r.mapping.get("oldbalanceOrg") == "solde_avant_source"
        assert r.mapping.get("newbalanceOrig") == "solde_apres_source"
        assert r.mapping.get("oldbalanceDest") == "solde_avant_dest"
        assert r.mapping.get("newbalanceDest") == "solde_apres_dest"

    def test_balance_english_variants(self):
        df = _paysim_renamed(
            balance_before="oldbalanceOrg",
            balance_after="newbalanceOrig",
            old_balance_dest="oldbalanceDest",
            new_balance_dest="newbalanceDest",
        )
        r = map_columns(df)
        # balance_before/after maps to source (no dest signal)
        assert r.mapping.get("oldbalanceOrg") == "balance_before"
        assert r.mapping.get("newbalanceOrig") == "balance_after"
        assert r.mapping.get("oldbalanceDest") == "old_balance_dest"
        assert r.mapping.get("newbalanceDest") == "new_balance_dest"

    # ── Pas de double affectation ──────────────────────────────────────────

    def test_no_column_used_twice(self):
        df = _paysim_renamed()
        r = map_columns(df)
        mapped_cols = list(r.mapping.values())
        assert len(mapped_cols) == len(set(mapped_cols)), "Une colonne affectée deux fois"

    # ── Confiance ─────────────────────────────────────────────────────────

    def test_exact_match_confidence_is_high(self):
        df = _paysim_renamed()
        r = map_columns(df)
        assert r.confidence["amount"] >= 0.90
        assert r.confidence["step"] >= 0.90

    # ── Type values normalization ─────────────────────────────────────────

    def test_type_values_uppercased(self):
        df = _paysim_renamed()
        df["type"] = ["transfer", "cash_out", "payment"]
        r = map_columns(df)
        assert set(r.df["type"].tolist()) == {"TRANSFER", "CASH_OUT", "PAYMENT"}

    def test_type_values_french(self):
        df = _paysim_renamed()
        df["type"] = ["virement", "retrait", "paiement"]
        r = map_columns(df)
        assert set(r.df["type"].tolist()) == {"TRANSFER", "CASH_OUT", "PAYMENT"}

    # ── Cas d'erreur ──────────────────────────────────────────────────────

    def test_missing_required_column(self):
        # CSV sans aucune colonne ressemblant à "amount"
        df = pd.DataFrame({
            "step": [1], "type": ["TRANSFER"],
            "oldbalanceOrg": [1000], "newbalanceOrig": [0],
            "oldbalanceDest": [0], "newbalanceDest": [1000],
            "nameOrig": ["C1"], "nameDest": ["M1"],
            # "amount" intentionnellement absent et aucun alias
            "completely_unrelated_column": [999],
        })
        r = map_columns(df)
        assert "amount" in r.unmapped

    def test_mapper_or_raise(self):
        df = pd.DataFrame({"col_x": [1], "col_y": [2]})
        with pytest.raises(ValueError, match="Colonnes requises"):
            ColumnMapper().map_or_raise(df)

    # ── Intégration avec l'API ─────────────────────────────────────────────

    def test_predict_with_renamed_columns(self, client, test_csv):
        """Le endpoint /api/predict accepte un CSV avec colonnes renommées."""
        import io
        df = pd.read_csv(io.BytesIO(test_csv))
        df = df.rename(columns={
            "amount": "montant",
            "type": "transaction_type",
            "oldbalanceOrg": "balance_before",
            "newbalanceOrig": "balance_after",
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/api/predict",
            files={"file": ("renamed.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_transactions"] > 0
        # Vérifier que le mapping est retourné
        assert "column_mapping" in data
        assert data["column_mapping"]["amount"]["original_name"] == "montant"
