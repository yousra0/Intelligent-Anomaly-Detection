"""
tests/test_feature_engineer.py
Tests du moteur de feature engineering générique.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineer import FeatureEngineer, engineer_features


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    types = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
    return pd.DataFrame({
        "step": np.arange(1, n + 1),
        "type": [types[i % 5] for i in range(n)],
        "amount": rng.uniform(100, 500_000, n),
        "nameOrig": [f"C{i % 10:04d}" for i in range(n)],   # 10 comptes sources
        "oldbalanceOrg": rng.uniform(100_000, 1_000_000, n),
        "newbalanceOrig": rng.uniform(0, 100_000, n),
        "nameDest": [f"M{i % 6:04d}" for i in range(n)],    # 6 comptes dest
        "oldbalanceDest": rng.uniform(0, 500_000, n),
        "newbalanceDest": rng.uniform(0, 500_000, n),
    })


def _datetime_step_df(n: int = 24) -> pd.DataFrame:
    base = pd.Timestamp("2024-06-10 09:00:00")
    df = _canonical_df(n)
    df["step"] = [base + pd.Timedelta(hours=i) for i in range(n)]
    return df


@pytest.fixture(scope="module")
def eng():
    return FeatureEngineer()


@pytest.fixture(scope="module")
def canon_df():
    return _canonical_df(60)


# ─────────────────────────────────────────────────────────────────────────────
# Tests temporels
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalFeatures:

    def test_three_temporal_features_generated(self, eng, canon_df):
        _, report = eng.engineer(canon_df)
        assert len(report.temporal_features) == 3
        assert "eng_is_weekend" in report.temporal_features
        assert "eng_is_business_hour" in report.temporal_features
        assert "eng_tx_day_of_week" in report.temporal_features

    def test_day_of_week_range(self, eng, canon_df):
        df_out, _ = eng.engineer(canon_df)
        dow = df_out["eng_tx_day_of_week"]
        assert dow.between(0, 6).all(), "day_of_week doit être entre 0 et 6"

    def test_is_weekend_binary(self, eng, canon_df):
        df_out, _ = eng.engineer(canon_df)
        vals = df_out["eng_is_weekend"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_is_business_hour_binary(self, eng, canon_df):
        df_out, _ = eng.engineer(canon_df)
        vals = df_out["eng_is_business_hour"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_weekend_and_business_hour_mutually_consistent(self, eng, canon_df):
        df_out, _ = eng.engineer(canon_df)
        # Si is_weekend=1, alors is_business_hour doit être 0
        mask = df_out["eng_is_weekend"] == 1.0
        assert (df_out.loc[mask, "eng_is_business_hour"] == 0.0).all()

    def test_datetime_step_extracted_correctly(self, eng):
        """step datetime → heure calendaire réelle."""
        n = 48
        base = pd.Timestamp("2024-06-10 00:00:00")  # lundi
        df = _canonical_df(n)
        # step = heure 0..47 (lun 0h → mer 23h)
        df["step"] = [base + pd.Timedelta(hours=i) for i in range(n)]
        df_out, report = eng.engineer(df)

        # Heure 0 du lundi doit être is_business_hour=0 (hors 9-18h)
        row_0 = df_out.iloc[0]
        assert row_0["eng_is_business_hour"] == 0.0  # 0h → hors horaires

        # Heure 10 du lundi doit être is_business_hour=1
        row_10 = df_out.iloc[10]
        assert row_10["eng_is_business_hour"] == 1.0  # 10h lundi → ouvert

        # Samedi (48h à partir du lundi = 2 jours → mercredi; 120h = vendredi)
        # step=120h → vendredi+24+24 = dimanche (day_of_week=6)
        # step=120: heure = 120%24=0h, day = 120//24=5 jours → vendredi+5=dim (5%7=5=sam)
        # Construire un df avec un step datetime samedi
        sat_df = pd.DataFrame([{
            "step": pd.Timestamp("2024-06-15 14:00"),  # samedi 14h
            "amount": 1000, "type": "TRANSFER",
            "nameOrig": "C0001", "oldbalanceOrg": 10000, "newbalanceOrig": 9000,
            "nameDest": "M0001", "oldbalanceDest": 0, "newbalanceDest": 1000,
        }])
        df_sat_out, _ = eng.engineer(sat_df)
        assert df_sat_out.iloc[0]["eng_is_weekend"] == 1.0
        assert df_sat_out.iloc[0]["eng_is_business_hour"] == 0.0

    def test_no_step_column_skips_temporal(self, eng):
        df = _canonical_df(10).drop(columns=["step"])
        _, report = eng.engineer(df)
        assert len(report.temporal_features) == 0
        assert any("temporelle" in s for s in report.skipped)

    def test_numeric_step_warning(self, eng, canon_df):
        _, report = eng.engineer(canon_df)
        assert any("cycliques" in w or "numérique" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de soldes
# ─────────────────────────────────────────────────────────────────────────────

class TestBalanceFeatures:

    def test_five_balance_features_generated(self, eng, canon_df):
        _, report = eng.engineer(canon_df)
        assert len(report.balance_features) == 5
        assert "eng_amount_ratio_src" in report.balance_features
        assert "eng_drain_pct_src" in report.balance_features
        assert "eng_balance_gap" in report.balance_features
        assert "eng_dest_gain_ratio" in report.balance_features
        assert "eng_drain_pct_dest" in report.balance_features

    def test_amount_ratio_src_range(self, eng):
        df = pd.DataFrame([{
            "step": 1, "type": "TRANSFER", "amount": 250_000,
            "nameOrig": "C001", "oldbalanceOrg": 500_000, "newbalanceOrig": 250_000,
            "nameDest": "M001", "oldbalanceDest": 0, "newbalanceDest": 250_000,
        }])
        df_out, _ = eng.engineer(df)
        # amount/oldbalanceOrg = 250000/500000 = 0.5
        assert df_out.iloc[0]["eng_amount_ratio_src"] == pytest.approx(0.5, rel=1e-4)

    def test_balance_gap_zero_for_consistent_transaction(self, eng):
        """Transaction cohérente : amount = oldbalanceOrg - newbalanceOrig."""
        df = pd.DataFrame([{
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C001", "oldbalanceOrg": 100_000, "newbalanceOrig": 0,
            "nameDest": "M001", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        }])
        df_out, _ = eng.engineer(df)
        assert df_out.iloc[0]["eng_balance_gap"] == pytest.approx(0.0, abs=1e-4)

    def test_balance_gap_nonzero_for_inconsistent(self, eng):
        """Incohérence comptable : amount ≠ variation de solde."""
        df = pd.DataFrame([{
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C001", "oldbalanceOrg": 100_000, "newbalanceOrig": 50_000,
            # Variation réelle = 50_000 ≠ amount=100_000
            "nameDest": "M001", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        }])
        df_out, _ = eng.engineer(df)
        assert df_out.iloc[0]["eng_balance_gap"] > 0.1

    def test_dest_gain_ratio_near_one_for_transfer(self, eng):
        """Pour un TRANSFER légitime, le dest reçoit exactement le montant envoyé."""
        df = pd.DataFrame([{
            "step": 1, "type": "TRANSFER", "amount": 50_000,
            "nameOrig": "C001", "oldbalanceOrg": 100_000, "newbalanceOrig": 50_000,
            "nameDest": "M001", "oldbalanceDest": 10_000, "newbalanceDest": 60_000,
        }])
        df_out, _ = eng.engineer(df)
        # (newDest - oldDest) / amount = 50_000 / 50_000 = 1.0
        assert df_out.iloc[0]["eng_dest_gain_ratio"] == pytest.approx(1.0, rel=1e-4)

    def test_drain_pct_src_100_percent(self, eng):
        """Compte source entièrement vidé."""
        df = pd.DataFrame([{
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C001", "oldbalanceOrg": 100_000, "newbalanceOrig": 0,
            "nameDest": "M001", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        }])
        df_out, _ = eng.engineer(df)
        assert df_out.iloc[0]["eng_drain_pct_src"] == pytest.approx(100.0, rel=1e-3)

    def test_no_balance_cols_skips_balance_features(self, eng):
        df = _canonical_df(10).drop(columns=["oldbalanceOrg", "newbalanceOrig",
                                               "oldbalanceDest", "newbalanceDest"])
        _, report = eng.engineer(df)
        assert len(report.balance_features) == 0
        assert any("balance" in s.lower() for s in report.skipped)

    def test_only_src_balance_generates_3_features(self, eng):
        df = _canonical_df(10).drop(columns=["oldbalanceDest", "newbalanceDest"])
        _, report = eng.engineer(df)
        # Source seulement → 3 features (ratio, drain, gap), pas dest
        src_features = [f for f in report.balance_features if "src" in f or "gap" in f]
        assert len(src_features) == 3
        assert "eng_dest_gain_ratio" not in report.balance_features

    def test_no_nan_in_balance_features(self, eng, canon_df):
        df_out, report = eng.engineer(canon_df)
        for feat in report.balance_features:
            assert not df_out[feat].isna().any(), f"NaN dans {feat}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests comportementaux
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralFeatures:

    def test_seven_behavioral_features_generated(self, eng, canon_df):
        _, report = eng.engineer(canon_df)
        assert len(report.behavioral_features) == 7
        expected = {
            "eng_orig_tx_count", "eng_orig_unique_dests",
            "eng_orig_avg_amount", "eng_orig_total_amount",
            "eng_orig_is_high_freq", "eng_dest_tx_count", "eng_dest_avg_received",
        }
        assert set(report.behavioral_features) == expected

    def test_tx_count_correct(self, eng):
        """Compte source 'C0000' apparaît 3 fois → orig_tx_count = 3."""
        df = pd.DataFrame([
            {"step": 1, "type": "TRANSFER", "amount": 100, "nameOrig": "C0000",
             "oldbalanceOrg": 1000, "newbalanceOrig": 900, "nameDest": "M0001",
             "oldbalanceDest": 0, "newbalanceDest": 100},
            {"step": 2, "type": "TRANSFER", "amount": 200, "nameOrig": "C0000",
             "oldbalanceOrg": 900, "newbalanceOrig": 700, "nameDest": "M0002",
             "oldbalanceDest": 0, "newbalanceDest": 200},
            {"step": 3, "type": "PAYMENT", "amount": 50, "nameOrig": "C0001",
             "oldbalanceOrg": 500, "newbalanceOrig": 450, "nameDest": "M0001",
             "oldbalanceDest": 0, "newbalanceDest": 50},
            {"step": 4, "type": "TRANSFER", "amount": 300, "nameOrig": "C0000",
             "oldbalanceOrg": 700, "newbalanceOrig": 400, "nameDest": "M0003",
             "oldbalanceDest": 0, "newbalanceDest": 300},
        ])
        df_out, _ = eng.engineer(df)
        c0000_rows = df_out[df_out["nameOrig"] == "C0000"]
        assert (c0000_rows["eng_orig_tx_count"] == 3).all()

    def test_unique_dests_correct(self, eng):
        """C0000 envoie vers M0001, M0002, M0003 → unique_dests = 3."""
        df = pd.DataFrame([
            {"step": i+1, "type": "TRANSFER", "amount": 100, "nameOrig": "C0000",
             "oldbalanceOrg": 1000, "newbalanceOrig": 900, "nameDest": f"M000{i}",
             "oldbalanceDest": 0, "newbalanceDest": 100}
            for i in range(3)
        ] + [
            {"step": 4, "type": "TRANSFER", "amount": 100, "nameOrig": "C0001",
             "oldbalanceOrg": 500, "newbalanceOrig": 400, "nameDest": "M0000",
             "oldbalanceDest": 0, "newbalanceDest": 100}
        ])
        df_out, _ = eng.engineer(df)
        c0000_rows = df_out[df_out["nameOrig"] == "C0000"]
        assert (c0000_rows["eng_orig_unique_dests"] == 3).all()
        c0001_rows = df_out[df_out["nameOrig"] == "C0001"]
        assert (c0001_rows["eng_orig_unique_dests"] == 1).all()

    def test_avg_amount_correct(self, eng):
        df = pd.DataFrame([
            {"step": 1, "type": "TRANSFER", "amount": 100.0, "nameOrig": "C0000",
             "oldbalanceOrg": 1000, "newbalanceOrig": 900, "nameDest": "M0001",
             "oldbalanceDest": 0, "newbalanceDest": 100},
            {"step": 2, "type": "TRANSFER", "amount": 300.0, "nameOrig": "C0000",
             "oldbalanceOrg": 900, "newbalanceOrig": 600, "nameDest": "M0002",
             "oldbalanceDest": 0, "newbalanceDest": 300},
        ])
        df_out, _ = eng.engineer(df)
        # moyenne de C0000 = (100+300)/2 = 200
        assert np.allclose(df_out["eng_orig_avg_amount"].values, 200.0, rtol=1e-3)

    def test_high_freq_flag(self, eng):
        """Comptes très actifs doivent être flaggés haute fréquence."""
        rng = np.random.default_rng(1)
        # C0000 fait 50 transactions, les 9 autres en font 1 chacun
        rows = [{"step": i, "type": "TRANSFER", "amount": 100, "nameOrig": "C0000",
                 "oldbalanceOrg": 1e6, "newbalanceOrig": 1e6-100,
                 "nameDest": f"M{i:04d}", "oldbalanceDest": 0, "newbalanceDest": 100}
                for i in range(50)]
        rows += [{"step": 50+i, "type": "PAYMENT", "amount": 50, "nameOrig": f"C{i+1:04d}",
                  "oldbalanceOrg": 1000, "newbalanceOrig": 950,
                  "nameDest": "M9999", "oldbalanceDest": 0, "newbalanceDest": 50}
                 for i in range(9)]
        df = pd.DataFrame(rows)
        df_out, _ = eng.engineer(df)
        # C0000 doit être high_freq
        c0000 = df_out[df_out["nameOrig"] == "C0000"]
        assert (c0000["eng_orig_is_high_freq"] == 1.0).all()

    def test_dest_tx_count(self, eng):
        """M0000 reçoit 3 transactions → dest_tx_count=3."""
        rows = [{"step": i+1, "type": "TRANSFER", "amount": 100, "nameOrig": f"C{i:04d}",
                 "oldbalanceOrg": 1000, "newbalanceOrig": 900, "nameDest": "M0000",
                 "oldbalanceDest": 0, "newbalanceDest": 100}
                for i in range(3)]
        rows.append({"step": 4, "type": "PAYMENT", "amount": 50, "nameOrig": "C0099",
                     "oldbalanceOrg": 500, "newbalanceOrig": 450, "nameDest": "M0001",
                     "oldbalanceDest": 0, "newbalanceDest": 50})
        df = pd.DataFrame(rows)
        df_out, _ = eng.engineer(df)
        m0000 = df_out[df_out["nameDest"] == "M0000"]
        assert (m0000["eng_dest_tx_count"] == 3).all()

    def test_no_nameOrig_skips_source_features(self, eng):
        df = _canonical_df(10).drop(columns=["nameOrig"])
        _, report = eng.engineer(df)
        assert "eng_orig_tx_count" not in report.behavioral_features

    def test_no_nameDest_skips_dest_features(self, eng):
        df = _canonical_df(10).drop(columns=["nameDest"])
        _, report = eng.engineer(df)
        assert "eng_dest_tx_count" not in report.behavioral_features

    def test_no_nan_in_behavioral_features(self, eng, canon_df):
        df_out, report = eng.engineer(canon_df)
        for feat in report.behavioral_features:
            assert not df_out[feat].isna().any(), f"NaN dans {feat}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests généraux
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineerGeneral:

    def test_total_15_features_on_full_paysim(self, eng, canon_df):
        _, report = eng.engineer(canon_df)
        assert report.n_generated == 15
        assert len(report.all_new_features) == 15

    def test_original_columns_preserved(self, eng, canon_df):
        df_out, _ = eng.engineer(canon_df)
        for col in canon_df.columns:
            assert col in df_out.columns

    def test_no_collision_with_eng_prefix(self, eng, canon_df):
        """Aucune colonne originale ne commence par 'eng_'."""
        original_eng_cols = [c for c in canon_df.columns if c.startswith("eng_")]
        assert len(original_eng_cols) == 0

    def test_all_dtypes_float32(self, eng, canon_df):
        df_out, report = eng.engineer(canon_df)
        for feat in report.all_new_features:
            assert df_out[feat].dtype == np.float32, f"{feat} dtype != float32"

    def test_no_inf_values(self, eng, canon_df):
        df_out, report = eng.engineer(canon_df)
        for feat in report.all_new_features:
            assert not np.isinf(df_out[feat].values).any(), f"Inf dans {feat}"

    def test_report_to_dict_serializable(self, eng, canon_df):
        import json
        _, report = eng.engineer(canon_df)
        json.dumps(report.to_dict())

    def test_empty_dataset_returns_zero_rows(self, eng):
        df = _canonical_df(0)
        df_out, report = eng.engineer(df)
        assert len(df_out) == 0
        # Pas de crash, rapport cohérent
        assert report.n_generated >= 0

    def test_single_row_no_crash(self, eng):
        df = _canonical_df(1)
        df_out, report = eng.engineer(df)
        assert len(df_out) == 1
        assert report.n_generated == 15


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration via l'API
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineerAPI:

    def test_predict_includes_feature_engineering(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "feature_engineering" in data
        fe = data["feature_engineering"]
        assert fe["n_generated"] >= 0
        assert "temporal_features" in fe
        assert "balance_features" in fe
        assert "behavioral_features" in fe

    def test_predict_feature_engineering_has_15_features(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        fe = data["feature_engineering"]
        assert fe["n_generated"] == 15

    def test_profile_includes_feature_engineering(self, client, test_csv):
        r = client.post(
            "/api/profile",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "feature_engineering" in data
        fe = data["feature_engineering"]
        assert fe["n_generated"] > 0

    def test_feature_engineering_specs_present(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        fe = data["feature_engineering"]
        assert "specs" in fe
        for feat in fe["all_new_features"]:
            assert feat in fe["specs"]
            spec = fe["specs"][feat]
            assert "formula" in spec
            assert "category" in spec
            assert spec["category"] in ("temporal", "balance", "behavioral")

    def test_renamed_columns_still_engineer_correctly(self, client):
        """CSV avec colonnes renommées → le mapping permet quand même l'engineering."""
        import io
        rng = np.random.default_rng(99)
        n = 15
        df = pd.DataFrame({
            "montant": rng.uniform(100, 100_000, n),
            "type_transaction": ["TRANSFER", "CASH_OUT", "PAYMENT"] * 5,
            "step": np.arange(1, n + 1),
            "oldbalanceOrg": rng.uniform(100_000, 1_000_000, n),
            "newbalanceOrig": rng.uniform(0, 100_000, n),
            "oldbalanceDest": rng.uniform(0, 500_000, n),
            "newbalanceDest": rng.uniform(0, 500_000, n),
            "nameOrig": [f"C{i:04d}" for i in range(n)],
            "nameDest": [f"M{i:04d}" for i in range(n)],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/api/predict",
            files={"file": ("renamed.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        fe = r.json()["feature_engineering"]
        assert fe["n_generated"] == 15
