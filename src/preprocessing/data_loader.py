import pandas as pd

def load_data(path: str, sample: bool = False, sample_size: int = 500000):
    """
    Charge le dataset PaySim depuis CSV ou Excel.
    

    Args:
        path: chemin vers le fichier
        sample: si True, charge un échantillon
        sample_size: taille de l'échantillon si sample=True

    Returns:
        DataFrame pandas
    """
    # détection de l'extension
    ext = path.split('.')[-1].lower()
    
    if ext == 'csv':
        reader = pd.read_csv
        kwargs = {'sep': ';', 'encoding': 'utf-8'}
    elif ext in ['xlsx', 'xls']:
        reader = pd.read_excel
        kwargs = {}  # pas de sep ni encoding pour Excel
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou Excel.")

    if sample:
        return reader(path, nrows=sample_size, **kwargs)
    return reader(path, **kwargs)