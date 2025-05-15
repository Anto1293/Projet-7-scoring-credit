# test :
# que les données sont bien découpées en ensembles d'entraînement et de validation.
# que la normalisation est effectuée correctement.
# que les dimensions des matrices retournées sont correctes.

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.split_and_scale import split_and_scale


@pytest.fixture
def sample_data():
    data = {
        'SK_ID_CURR': [1, 2, 3, 4, 5, 6, 7, 8],
        'TARGET':     [0, 1, 0, 1, 0, 1, 0, 1],
        'NUMERIC_FEATURE_1': [10, 20, 30, 40, 50, 60, 70, 80],
        'NUMERIC_FEATURE_2': [5, 15, 25, 35, 45, 55, 65, 75],
    }
    df = pd.DataFrame(data)
    return df

def test_split_and_scale(sample_data):
    df = sample_data
    target_col = 'TARGET'
    numerical_features = ['NUMERIC_FEATURE_1', 'NUMERIC_FEATURE_2']
    
    # Exécuter la fonction
    X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, scaler = split_and_scale(df, target_col, numerical_features)
    
    # Vérifier les dimensions
    assert X_train.shape[0] == 6 # 75% de 8 échantillons (train_échantillon)
    assert X_val.shape[0] == 2  # 25% de 8 échantillons (val échantillon)
    
    # Vérifier que les caractéristiques numériques sont normalisées
    assert not np.allclose(X_train[numerical_features].mean(), 0)  # Non normalisé avant transformation
    assert np.allclose(X_train_scaled[numerical_features].mean(), 0, atol=0.2)  # Moyenne autour de 0  (atol) après normalisation car petit échantillon
    
    # Vérifier que les données sont bien scalées
    assert isinstance(scaler, StandardScaler)