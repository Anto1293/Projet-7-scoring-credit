import numpy as np

def clean_app_data(df):
    """
    Nettoie les valeurs aberrantes
    """
    df = df.copy()
    df.loc[df['CNT_CHILDREN'] >= 3, 'CNT_CHILDREN'] = 3
    df['IS_UNEMPLOYED'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df = df[df['CODE_GENDER'] != 'XNA']
    return df

def clean_previous_application(df):
    cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in cols:
        df[col] = df[col].replace(365243, np.nan)
    return df

def process_missing_values(df):
    """
    Nettoie les valeurs manquantes par Uknown ou par la médiane
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df
