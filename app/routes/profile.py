"""
app/routes/profile.py
POST /api/profile — Analyse d'un CSV sans prédiction
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.auth.dependencies import CurrentUser, get_current_user
from app.services.column_mapper import map_columns
from app.services.dataset_profiler import profile_dataset
from app.services.feature_engineer import engineer_features

router = APIRouter()


@router.post("/profile")
async def profile_csv(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Analyse un fichier CSV et retourne le rapport de profilage complet :
    - Types sémantiques de chaque colonne
    - Valeurs manquantes
    - Cardinalité
    - Stats numériques / catégorielles
    - Détection de colonnes quasi-constantes, identifiants, dates
    - Score de qualité global
    - Recommandations
    - Mapping sémantique vers les colonnes canoniques
    """
    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le CSV : {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne de données.")

    # Profilage
    profile = profile_dataset(df)

    # Mapping sémantique
    mapping_result = map_columns(df)
    mapping_summary = {
        canonical: {
            "original_name": orig,
            "confidence": round(mapping_result.confidence.get(canonical, 0), 2),
        }
        for canonical, orig in mapping_result.mapping.items()
    }

    # Feature engineering (non-bloquant)
    eng_report = None
    if mapping_result.success:
        try:
            _, eng_report = engineer_features(
                mapping_result.df, profile=profile, mapping=mapping_result.mapping
            )
        except Exception:
            pass

    response = profile.to_dict()
    response["semantic_mapping"] = {
        "mapping": mapping_summary,
        "unmapped_required": mapping_result.unmapped,
        "warnings": mapping_result.warnings,
        "ready_for_prediction": mapping_result.success,
    }
    if eng_report is not None:
        response["feature_engineering"] = eng_report.to_dict()

    return response
