"""
app/routes/predict.py
POST /api/predict — Analyse + prédiction de fraude sur un CSV

Pipeline
────────
1. Lecture CSV
2. Profilage sémantique (DatasetProfiler)
3. Mapping des colonnes (ColumnMapper)
4. Détection du mode de prédiction (SchemaDetector)
   → "paysim"      : XGBoost + Autoencoder  (schéma PaySim complet)
   → "ae_isoforest": Autoencoder + IsoForest (schéma partiel, amount détecté)
   → "ae_only"     : Autoencoder seul        (amount détecté, peu de numériques)
   → "isoforest"   : IsoForest seul          (schéma inconnu, colonnes numériques)
5. Construction dynamique des features (DynamicFeatureBuilder, avec fallbacks)
6. Feature engineering générique (FeatureEngineer)
7. Prédiction selon le mode détecté
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from app.services.column_mapper import map_columns
from app.services.dataset_profiler import profile_dataset
from app.services.feature_builder import build_features_dynamic
from app.services.feature_engineer import engineer_features
from app.services.generic_predictor import predict_generic_batch
from app.services.predictor import predict_batch
from app.services.schema_detector import detect_schema_mode

router = APIRouter()


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    models = request.app.state.models

    # ── 1. Lecture CSV ──────────────────────────────────────────────────────
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le CSV : {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne de données.")

    # ── 2. Profilage ────────────────────────────────────────────────────────
    dataset_profile = None
    try:
        dataset_profile = profile_dataset(df)
    except Exception as e:
        request.app.state._last_profile_error = str(e)

    # ── 3. Mapping sémantique ───────────────────────────────────────────────
    mapping_result = map_columns(df)

    # ── 4. Détection du mode de prédiction ─────────────────────────────────
    try:
        schema_result = detect_schema_mode(mapping_result, df)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "columns_in_csv": list(df.columns),
                "partial_mapping": {
                    k: {"original": v, "confidence": round(mapping_result.confidence.get(k, 0), 2)}
                    for k, v in mapping_result.mapping.items()
                },
            },
        )

    df_mapped = mapping_result.df   # colonnes renommées (peut être partiel)

    # ── 5. Construction des features ────────────────────────────────────────
    X_arr = None
    build_report = None
    if schema_result.use_ae or schema_result.use_xgb:
        try:
            X_arr, build_report = build_features_dynamic(
                df_mapped, dataset_profile, models["scaler"]
            )
        except Exception as e:
            if schema_result.mode == "paysim":
                raise HTTPException(
                    status_code=500, detail=f"Erreur construction features : {e}"
                )
            # En mode générique l'AE est optionnel
            schema_result.warnings.append(f"Construction features AE échouée : {e}")
            schema_result.use_ae = False

    # ── 6. Feature engineering générique ───────────────────────────────────
    df_enriched = df_mapped
    eng_report = None
    try:
        df_enriched, eng_report = engineer_features(
            df_mapped, profile=dataset_profile, mapping=mapping_result.mapping
        )
    except Exception as e:
        schema_result.warnings.append(f"Feature engineering ignoré : {e}")

    # ── 7. Prédiction ───────────────────────────────────────────────────────
    try:
        if schema_result.mode == "paysim":
            transactions = predict_batch(X_arr, models, df_mapped)
        else:
            transactions = predict_generic_batch(
                df_original=df,
                X_arr=X_arr,
                models=models,
                schema_result=schema_result,
                df_mapped=df_mapped,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {e}")

    # ── Cache pour /api/explain ─────────────────────────────────────────────
    request.app.state.results_cache = {
        "X_arr": X_arr,
        "df": df_enriched,
        "transactions": transactions,
    }

    # ── Résumé ───────────────────────────────────────────────────────────────
    frauds = [t for t in transactions if t["is_fraud_predicted"]]
    amount_at_risk = sum(t["amount"] for t in frauds)
    fraud_rate_pct = round(len(frauds) / max(len(transactions), 1) * 100, 4)

    if schema_result.mode == "paysim":
        threshold = models["thresholds"].get("XGB_smote", 0.355)
    else:
        threshold = None

    column_mapping_info = {
        k: {
            "original_name": v,
            "confidence": round(mapping_result.confidence.get(k, 0), 2),
        }
        for k, v in mapping_result.mapping.items()
    }

    response: dict = {
        "n_transactions": len(transactions),
        "n_fraud": len(frauds),
        "fraud_rate_pct": fraud_rate_pct,
        "amount_at_risk": round(amount_at_risk, 2),
        "model_used": schema_result.model_label,
        "prediction_mode": schema_result.mode,
        "schema_detection": schema_result.to_dict(),
        "column_mapping": column_mapping_info,
        "mapping_warnings": mapping_result.warnings,
        "feature_engineering": eng_report.to_dict() if eng_report else {},
        "transactions": transactions,
    }

    if threshold is not None:
        response["threshold"] = threshold

    if build_report is not None:
        response["feature_build"] = build_report.to_dict()

    if dataset_profile is not None:
        response["dataset_profile"] = {
            "n_rows": dataset_profile.n_rows,
            "n_cols": dataset_profile.n_cols,
            "global_quality_score": round(dataset_profile.global_quality_score, 1),
            "numeric_cols": dataset_profile.numeric_cols,
            "categorical_cols": dataset_profile.categorical_cols,
            "datetime_cols": dataset_profile.datetime_cols,
            "identifier_cols": dataset_profile.identifier_cols,
            "quasi_constant_cols": dataset_profile.quasi_constant_cols,
            "high_missing_cols": dataset_profile.high_missing_cols,
            "recommendations": dataset_profile.recommendations,
            "profiling_time_ms": round(dataset_profile.profiling_time_ms, 1),
        }

    return response
