### Vérification des valeurs manquantes
import pandas as pd

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