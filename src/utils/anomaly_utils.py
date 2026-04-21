### Vérification des valeurs manquantes
import pandas as pd
import numpy as np

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