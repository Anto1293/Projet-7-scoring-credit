import pandas as pd

def one_hot_encoder(df, nan_as_category=True):
     """
    Encode les colonnes catégorielles du DataFrame en one-hot encoding.
    
    Args:
        df (pd.DataFrame): Le DataFrame à encoder.
        nan_as_category (bool): Si True, les valeurs manquantes seront traitées comme une catégorie distincte.

    Returns:
        df (pd.DataFrame): Le DataFrame avec les colonnes one-hot encodées.
        original_columns (list): La liste des colonnes qui ont été encodées.
    """
    original_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df = pd.get_dummies(df, columns=original_columns, dummy_na=nan_as_category)
    return df, original_columns


def encode_categoricals(df):
    df = df.copy()

    # 1. Label Encoding manuel pour binaire
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 0, 'F': 1})
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1})

    # 2. get_dummies pour toutes les colonnes catégorielles restantes
    cat_cols = [
        'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
        'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True, drop_first=True)

    return df