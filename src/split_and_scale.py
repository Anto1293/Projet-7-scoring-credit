from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def split_and_scale(df: pd.DataFrame, target_col: str, numerical_features: list, test_size: float = 0.2, random_state: int = 42):
    """
    Split les données en train/val et applique la normalisation sur les colonnes numériques.
    Retourne les splits normalisés et non normalisés.
    """
    X = df.drop(columns=[target_col, 'SK_ID_CURR'])
    y = df[target_col]
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    scaler.fit(X[numerical_features])
    
    X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])
    X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])

    return X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, scaler
