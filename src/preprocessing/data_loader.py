### Importation de données

from pathlib import Path
import pandas as pd


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read CSV while handling common delimiters (, ; \t)."""
    # Fast path: standard comma-separated CSV (PaySim default)
    df = pd.read_csv(path, encoding="utf-8")

    # If parsing produced a single merged column, retry with alternate delimiters.
    if len(df.columns) == 1:
        first_col = str(df.columns[0])
        if ";" in first_col:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        elif "\t" in first_col:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")

    return df

def load_data(path, sample: bool = False, sample_size: int = 200_000):
    
    """
    Charge le dataset PaySim depuis CSV ou Excel.
    

    Args:
        path: chemin vers le fichier
        sample: si True, retourne un échantillon aléatoire
        sample_size: taille de l'échantillon si sample=True

    Returns:
        DataFrame pandas
    """
    
    path = Path(path)
    
    
    # détection de l'extension
    ext = path.suffix.lower().replace('.', '')
    
    if ext == 'csv':
        reader = _read_csv_with_fallback
        kwargs = {}
    elif ext in ['xlsx', 'xls']:
        reader = pd.read_excel
        kwargs = {}
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou Excel.")

    df = reader(path, **kwargs)

    if sample:
        n = min(int(sample_size), len(df))
        return df.sample(n=n, random_state=42)

    return df