import numpy as np
import pandas as pd

REQUIRED = ["X","Y","Z","R"]

def load_points_csv(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    df = df.dropna(subset=REQUIRED).copy()
    # Nettoyages usuels
    df = df.drop_duplicates(subset=["X","Y","Z"])
    # Option: enlever R<=0 si bruit/saturation
    return df
