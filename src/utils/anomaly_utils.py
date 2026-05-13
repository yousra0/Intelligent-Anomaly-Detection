### Vérification des valeurs manquantes
import pandas as pd
import numpy as np
from pathlib import Path


def validate_score_array(scores, *, name: str = "scores") -> dict:
    """Validate anomaly score arrays before saving or loading them."""
    arr = np.asarray(scores)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")

    min_value = float(arr.min())
    max_value = float(arr.max())
    if min_value < 0.0:
        raise ValueError(f"{name} must be non-negative, got min {min_value}.")

    return {
        "name": name,
        "shape": tuple(arr.shape),
        "min": min_value,
        "max": max_value,
        "mean": float(arr.mean()),
    }


def validate_saved_score_file(file_path: Path | str, *, name: str = "scores") -> dict:
    """Load and validate a persisted score array."""
    path = Path(file_path)
    scores = np.load(path, allow_pickle=True)
    return validate_score_array(scores, name=name)

def check_missing_values(df):
    """
    Retourne le nombre et le pourcentage de valeurs manquantes.
    """
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    
    return pd.DataFrame({
        "Missing Values": missing,
        
        "Percentage": percent
    }).sort_values(by="Percentage", ascending=False)


def check_class_imbalance(df: pd.DataFrame, target: str) -> dict:
    """Return class counts, percentages, and imbalance ratio for target."""
    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est introuvable.")

    counts = df[target].value_counts(dropna=False).to_dict()
    total = len(df)
    pct = {k: (v / total) * 100 for k, v in counts.items()}

    major = max(counts.values()) if counts else 0
    minor = min(counts.values()) if counts else 0
    ratio = (major / minor) if minor else np.inf

    return {
        "target": target,
        "total_rows": total,
        "counts": counts,
        "percentages": {k: round(v, 6) for k, v in pct.items()},
        "imbalance_ratio_major_to_minor": float(ratio),
    }


def detect_balance_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where origin balance delta does not match amount transferred."""
    required_cols = {"oldbalanceOrg", "newbalanceOrig", "amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la detection: {sorted(missing)}")

    out = df.copy()
    out["observed_delta_orig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balance_gap_vs_amount"] = out["observed_delta_orig"] - out["amount"]
    out["is_balance_anomaly"] = (~np.isclose(out["observed_delta_orig"], out["amount"]))
    return out[out["is_balance_anomaly"]].copy()