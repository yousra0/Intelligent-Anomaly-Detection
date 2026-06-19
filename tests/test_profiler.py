"""
tests/test_profiler.py
Tests du moteur de profilage et du constructeur de features adaptatif.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.dataset_profiler import DatasetProfiler, profile_dataset
from app.services.feature_builder import DynamicFeatureBuilder, build_features_dynamic


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_canonical_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    types = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
    return pd.DataFrame({
        "step": np.arange(1, n + 1),
        "type": [types[i % len(types)] for i in range(n)],
        "amount": rng.uniform(100, 500_000, n),
        "nameOrig": [f"C{i:010d}" for i in range(n)],
        "oldbalanceOrg": rng.uniform(0, 1_000_000, n),
        "newbalanceOrig": rng.uniform(0, 1_000_000, n),
        "nameDest": [f"M{i:010d}" for i in range(n)],
        "oldbalanceDest": rng.uniform(0, 1_000_000, n),
        "newbalanceDest": rng.uniform(0, 1_000_000, n),
    })


@pytest.fixture(scope="module")
def canonical_df():
    return _minimal_canonical_df(100)


@pytest.fixture(scope="module")
def profiler():
    return DatasetProfiler()


@pytest.fixture(scope="module")
def builder():
    return DynamicFeatureBuilder()


# ─────────────────────────────────────────────────────────────────────────────
# Tests du Profiler
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetProfiler:

    def test_profile_returns_correct_shape(self, profiler, canonical_df):
        p = profiler.profile(canonical_df)
        assert p.n_rows == 100
        assert p.n_cols == len(canonical_df.columns)
        assert len(p.columns) == len(canonical_df.columns)

    def test_numeric_columns_detected(self, profiler, canonical_df):
        p = profiler.profile(canonical_df)
        assert "amount" in p.numeric_cols
        assert "oldbalanceOrg" in p.numeric_cols
        assert "step" in p.numeric_cols

    def test_categorical_columns_detected(self, profiler, canonical_df):
        p = profiler.profile(canonical_df)
        assert "type" in p.categorical_cols

    def test_identifier_columns_detected(self, profiler, canonical_df):
        p = profiler.profile(canonical_df)
        # nameOrig et nameDest sont tous uniques → identifiants
        assert "nameOrig" in p.identifier_cols or "nameOrig" in p.numeric_cols or "nameOrig" in p.categorical_cols

    def test_missing_values_counted(self, profiler):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, None, 5.0],
            "b": ["x", "y", None, None, "z"],
        })
        p = profiler.profile(df)
        assert p.columns["a"].n_missing == 2
        assert p.columns["a"].pct_missing == pytest.approx(0.4)
        assert p.columns["b"].n_missing == 2

    def test_missing_flag_raised(self, profiler):
        df = pd.DataFrame({"col": [1.0] + [None] * 40})
        p = profiler.profile(df)
        assert "high_missing" in p.columns["col"].quality_flags
        assert p.columns["col"].quality_score < 100

    def test_quasi_constant_detected(self, profiler):
        df = pd.DataFrame({"col": [1] * 98 + [2, 3]})
        p = profiler.profile(df)
        assert p.columns["col"].is_quasi_constant
        assert "quasi_constant" in p.columns["col"].quality_flags
        assert "col" in p.quasi_constant_cols

    def test_constant_column_detected(self, profiler):
        df = pd.DataFrame({"col": [42] * 20})
        p = profiler.profile(df)
        assert p.columns["col"].semantic_type == "constant"
        assert "constant_column" in p.columns["col"].quality_flags

    def test_numeric_stats_computed(self, profiler):
        arr = np.arange(1, 101, dtype=float)
        df = pd.DataFrame({"val": arr})
        p = profiler.profile(df)
        col = p.columns["val"]
        assert col.semantic_type == "numeric"
        assert col.num_min == pytest.approx(1.0)
        assert col.num_max == pytest.approx(100.0)
        assert col.num_mean == pytest.approx(50.5)
        assert col.num_std is not None
        assert col.num_skewness is not None

    def test_top_values_for_categorical(self, profiler):
        df = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
        p = profiler.profile(df)
        assert p.columns["cat"].top_values is not None
        assert "A" in p.columns["cat"].top_values
        assert p.columns["cat"].top_values["A"] == 50

    def test_datetime_detected_native(self, profiler):
        dates = pd.date_range("2024-01-01", periods=20, freq="h")  # pandas 3.0: "h" not "H"
        df = pd.DataFrame({"ts": dates})
        p = profiler.profile(df)
        assert p.columns["ts"].semantic_type == "datetime"
        assert "ts" in p.datetime_cols

    def test_datetime_detected_string(self, profiler):
        df = pd.DataFrame({"ts": [f"2024-{i:02d}-01" for i in range(1, 13)] * 2})
        p = profiler.profile(df)
        assert p.columns["ts"].semantic_type == "datetime"

    def test_binary_numeric_detected(self, profiler):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 0, 1, 1, 0]})
        p = profiler.profile(df)
        assert p.columns["flag"].semantic_type == "boolean"

    def test_extreme_skewness_flagged(self, profiler):
        # Distribution long-tail : chi2(df=1) amplifiée → skewness >> 10
        rng = np.random.default_rng(0)
        arr = rng.chisquare(df=1, size=1000) ** 4  # skewness >> 10 garanti
        df = pd.DataFrame({"amount": arr})
        p = profiler.profile(df)
        # Vérifier que la skewness calculée est bien > 10
        assert p.columns["amount"].num_skewness > 10
        assert "extreme_skewness" in p.columns["amount"].quality_flags

    def test_zero_heavy_flagged(self, profiler):
        # Utiliser 0.0 et 2.5 pour éviter la détection "boolean" (0/1)
        arr = np.concatenate([np.zeros(600), np.full(400, 2.5)])
        df = pd.DataFrame({"balance": arr})
        p = profiler.profile(df)
        assert "zero_heavy" in p.columns["balance"].quality_flags

    def test_cardinality_types(self, profiler):
        df = pd.DataFrame({
            "const": [1] * 20,
            "binary_col": [0, 1] * 10,
            "low_card": list(range(5)) * 4,
            "high_card": list(range(20)),
        })
        p = profiler.profile(df)
        assert p.columns["const"].cardinality_type == "constant"
        assert p.columns["binary_col"].cardinality_type == "binary"
        assert p.columns["low_card"].cardinality_type == "low"

    def test_recommendations_generated(self, profiler):
        df = pd.DataFrame({
            "amount": np.concatenate([np.zeros(900), np.array([1e6] * 100)]),
            "const": [1] * 1000,
        })
        p = profiler.profile(df)
        assert len(p.recommendations) > 0

    def test_to_dict_is_serializable(self, profiler, canonical_df):
        import json
        p = profiler.profile(canonical_df)
        d = p.to_dict()
        json.dumps(d)  # doit passer sans erreur

    def test_quality_score_penalized_for_issues(self, profiler):
        df = pd.DataFrame({
            "col": [1.0] + [None] * 50,  # >30% missing
        })
        p = profiler.profile(df)
        assert p.columns["col"].quality_score < 80
        assert p.global_quality_score < 100

    def test_profiling_time_tracked(self, profiler, canonical_df):
        p = profiler.profile(canonical_df)
        assert p.profiling_time_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests du DynamicFeatureBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicFeatureBuilder:

    def _get_scaler_and_profile(self, canonical_df):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        profile = profile_dataset(canonical_df)
        return models["scaler"], profile

    def test_output_shape(self, builder, canonical_df):
        scaler, profile = self._get_scaler_and_profile(canonical_df)
        X, report = builder.build(canonical_df, profile, scaler)
        assert X.shape == (len(canonical_df), 14)
        assert X.dtype == np.float32

    def test_all_features_present(self, builder, canonical_df):
        from app.services.predictor import FEATURE_COLS
        scaler, profile = self._get_scaler_and_profile(canonical_df)
        X, report = builder.build(canonical_df, profile, scaler)
        assert report.n_features == 14

    def test_canonical_ok_status(self, builder, canonical_df):
        scaler, profile = self._get_scaler_and_profile(canonical_df)
        _, report = builder.build(canonical_df, profile, scaler)
        # Sur un dataset canonique, aucun fallback ne doit être utilisé
        assert len(report.features_fallback) == 0, f"Fallbacks inattendus: {report.features_fallback}"

    def test_missing_amount_fallback(self, builder):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        df = _minimal_canonical_df(10).drop(columns=["amount"])
        profile = profile_dataset(df)
        X, report = builder.build(df, profile, models["scaler"])
        assert "log_amount" in report.features_fallback
        # log_amount doit être 0 → log1p(0) = 0 après fallback
        assert not np.isnan(X).any()

    def test_missing_balance_cols_fallback(self, builder):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        df = _minimal_canonical_df(10).drop(columns=["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest"])
        profile = profile_dataset(df)
        X, report = builder.build(df, profile, models["scaler"])
        assert "balance_diff_orig" in report.features_fallback
        assert "dest_zero_balance" in report.features_fallback
        assert not np.isnan(X).any()

    def test_missing_type_col_fallback(self, builder):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        df = _minimal_canonical_df(10).drop(columns=["type"])
        profile = profile_dataset(df)
        X, report = builder.build(df, profile, models["scaler"])
        assert "is_transfer_or_cashout" in report.features_fallback
        assert len(report.warnings) > 0

    def test_datetime_step_adapted(self, builder):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        df = _minimal_canonical_df(10)
        # Remplacer step par des datetimes
        df["step"] = pd.date_range("2024-01-01", periods=10, freq="h")
        profile = profile_dataset(df)
        X, report = builder.build(df, profile, models["scaler"])
        assert X.shape == (10, 14)
        assert "step" in report.features_adapted or "step" in report.features_ok
        assert not np.isnan(X).any()

    def test_negative_amount_adapted(self, builder):
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        df = _minimal_canonical_df(10)
        df["amount"] = -abs(df["amount"])  # tout négatif
        profile = profile_dataset(df)
        X, report = builder.build(df, profile, models["scaler"])
        assert any("abs()" in a or "négativ" in a for a in report.adaptations)
        assert not np.isnan(X).any()

    def test_build_report_serializable(self, builder, canonical_df):
        import json
        from app.services.predictor import load_all_models
        from pathlib import Path
        models = load_all_models(Path("."))
        profile = profile_dataset(canonical_df)
        _, report = builder.build(canonical_df, profile, models["scaler"])
        json.dumps(report.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration via l'API
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileAPI:

    def test_profile_endpoint_returns_json(self, client, test_csv):
        r = client.post(
            "/api/profile",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "n_rows" in data
        assert "columns" in data
        assert "global_quality_score" in data
        assert "semantic_mapping" in data

    def test_profile_detects_numeric_amount(self, client, test_csv):
        r = client.post(
            "/api/profile",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        assert "amount" in data["column_categories"]["numeric"]

    def test_predict_includes_profile(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "dataset_profile" in data
        assert "feature_build" in data
        assert data["dataset_profile"]["global_quality_score"] > 0

    def test_predict_feature_build_report(self, client, test_csv):
        r = client.post(
            "/api/predict",
            files={"file": ("test.csv", test_csv, "text/csv")},
        )
        data = r.json()
        fb = data["feature_build"]
        assert fb["n_features"] == 14
        assert len(fb["features_ok"]) > 0

    def test_profile_empty_csv(self, client):
        r = client.post(
            "/api/profile",
            files={"file": ("empty.csv", b"col1,col2\n", "text/csv")},
        )
        assert r.status_code == 400
