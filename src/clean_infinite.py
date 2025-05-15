import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les valeurs infinies et NaNs, puis applique une imputation médiane.
    """
    print(" Vérification initiale des NaNs et valeurs infinies :")
    print(df.isnull().mean())
    print(np.isinf(df).mean())

    # Remplacer inf par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    print("\nVérification post-imputation :")
    print(df_imputed.isnull().sum())
    print(np.isinf(df_imputed).sum())

    return df_imputed
