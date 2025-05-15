import pandas as pd
import numpy as np
import pytest
from src.clean_infinite import clean_and_impute

def test_clean_and_impute_removes_nans_and_infs():
    # Création d’un DataFrame avec NaNs et infinis
    df = pd.DataFrame({
        'A': [1, 2, np.nan, np.inf],
        'B': [5, np.nan, 7, -np.inf],
        'C': [np.inf, 6, np.nan,8]
    })

    # Application de la fonction
    df_cleaned = clean_and_impute(df)

    # Vérification : plus de NaN ni d'infini
    assert not df_cleaned.isnull().any().any(), "Le DataFrame contient encore des NaNs"
    assert not np.isinf(df_cleaned.values).any(), "Le DataFrame contient encore des valeurs infinies"

    # Vérification des dimensions : mêmes colonnes, mêmes lignes
    assert df_cleaned.shape == df.shape, "Les dimensions du DataFrame ont changé"
