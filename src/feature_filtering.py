def correlation_with_target(df, target_col='TARGET', normality_alpha=0.05):
    """
    Fonction pour calculer la corrélation des variables numériques avec la variable cible ('TARGET').
    Si la distribution de la variable est normale, on utilise Pearson, sinon Spearman.
    """
    results = []
    numerical_cols = df.select_dtypes(include=['number', 'bool']).columns.drop(target_col, errors='ignore')

    for col in numerical_cols:
        x = df[col].astype(float)
        y = df[target_col]

        if x.nunique() <= 1:
            continue

        stat, p = normaltest(x.dropna())
        is_normal = p > normality_alpha

        try:
            if is_normal:
                corr, _ = pearsonr(x.dropna(), y.loc[x.dropna().index])
            else:
                corr, _ = spearmanr(x.dropna(), y.loc[x.dropna().index])
        except Exception:
            corr = np.nan

        results.append({'variable': col, 'correlation': corr})

    corr_df = pd.DataFrame(results)
    corr_df['abs_corr'] = corr_df['correlation'].abs()

    return corr_df.sort_values(by='abs_corr', ascending=False)

def regroup_organization_types(df):
    """
    Regroupe les variables relatives à 'ORGANIZATION_TYPE' en catégories plus larges pour une meilleure interprétation.
    """
    org_cols = [col for col in df.columns if col.startswith('ORGANIZATION_TYPE_')]
    groupes = {
        'SECTOR_Industry': [col for col in org_cols if 'Industry' in col],
        'SECTOR_Trade': [col for col in org_cols if 'Trade' in col],
        'SECTOR_Transport': [col for col in org_cols if 'Transport' in col],
        'SECTOR_Business Entity': [col for col in org_cols if 'Business Entity' in col],
        'SECTOR_Government': [col for col in org_cols if 'Government' in col],
        'SECTOR_Security': [col for col in org_cols if 'Security' in col],
        'SECTOR_Services': ['ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Restaurant'],
        'SECTOR_Construction': ['ORGANIZATION_TYPE_Construction'],
        'SECTOR_Medicine': ['ORGANIZATION_TYPE_Medicine'],
        'SECTOR_Police': ['ORGANIZATION_TYPE_Police'],
    }
    cols_deja_groupées = sum(groupes.values(), [])
    groupes['SECTOR_Other'] = [col for col in org_cols if col not in cols_deja_groupées]

    for group_name, columns in groupes.items():
        df[group_name] = df[columns].max(axis=1)

    return df.drop(columns=org_cols)

def regroup_occupation_types(df):
    """
    Regroupe les variables relatives à 'OCCUPATION_TYPE' en catégories plus larges pour une meilleure interprétation.
    """
    occupation_cols = [col for col in df.columns if col.startswith('OCCUPATION_TYPE_')]
    groupes = {
    'OCCUPATION_Labor_Work': ['OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Drivers'],
    'OCCUPATION_Sales_Services': ['OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Waiters/barmen staff','OCCUPATION_TYPE_Private service staff'],
    'OCCUPATION_Medical_Staff': ['OCCUPATION_TYPE_Medicine staff'],
    'OCCUPATION_Security': ['OCCUPATION_TYPE_Security staff'],
    'OCCUPATION_Management_Core': ['OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Core staff'],
    'OCCUPATION_Other': ['OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Unknown']}

    for group_name, columns in groupes.items():
        df[group_name] = df[columns].max(axis=1)

    return df.drop(columns=occupation_cols)

# -------------------------------------------------------------------------------------------------
# --- Nettoyage des données (train et test) ------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def clean_dataset(df, target_col='TARGET', target_present=True):
    """
    Cette fonction effectue le nettoyage des données en supprimant les colonnes peu pertinentes, 
    en regroupant certaines variables et en vérifiant la cohérence des données.
    """
    
    df_clean = df.copy()
    
    # Si le DataFrame contient la cible, on la retire pour le nettoyage
    if target_present and target_col in df_clean.columns:
        target_values = df_clean[target_col]
        df_clean = df_clean.drop(columns=[target_col])
        

    # 1. Suppression des colonnes très peu corrélées avec la target
    low_corr_vars = [
    'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'LIVE_REGION_NOT_WORK_REGION',
    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_MODE',
    'YEARS_BEGINEXPLUATATION_MEDI', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
    'BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'POS_MONTHS_BALANCE_MAX', 'INSTAL_PAYMENT_PERC_MEAN',
    'INSTAL_DPD_MAX', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN', 'CC_SK_DPD_MEAN', 'INCOME_CREDIT_PERC',
    'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B',
    'NAME_TYPE_SUITE_Spouse, partner', 'NAME_INCOME_TYPE_Maternity leave',
    'NAME_INCOME_TYPE_Student', 'NAME_EDUCATION_TYPE_Incomplete higher',
    'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Unknown',
    'NAME_HOUSING_TYPE_Municipal apartment', 'OCCUPATION_TYPE_HR staff',
    'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
    'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY',
    'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY',
    'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'ORGANIZATION_TYPE_Business Entity Type 1',
    'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Cleaning',
    'ORGANIZATION_TYPE_Culture', 'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency',
    'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 10',
    'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Industry: type 13',
    'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 4',
    'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Industry: type 6',
    'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Industry: type 8',
    'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Mobile',
    'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Postal', 'ORGANIZATION_TYPE_Realtor',
    'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom',
    'ORGANIZATION_TYPE_Trade: type 1', 'ORGANIZATION_TYPE_Trade: type 2',
    'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5',
    'ORGANIZATION_TYPE_Transport: type 1', 'ORGANIZATION_TYPE_Transport: type 2',
    'FONDKAPREMONT_MODE_not specified', 'HOUSETYPE_MODE_terraced house',
    'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Others', 'EMERGENCYSTATE_MODE_Yes'
]

    
    df_clean = df_clean.drop(columns=[col for col in low_corr_vars if col in df_clean.columns])

    # 2. Suppression des variables très corrélées entre elles et peu pertinentes pour le scoring
    highly_corr_vars = [
    'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY','REG_REGION_NOT_LIVE_REGION', 
    'REGION_RATING_CLIENT', 'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    'APARTMENTS_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 
    'YEARS_BUILD_MODE', 'COMMONAREA_AVG', 'COMMONAREA_MODE', 'ELEVATORS_AVG', 'ELEVATORS_MODE', 'ENTRANCES_AVG', 
    'ENTRANCES_MODE', 'FLOORSMIN_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MODE', 
    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 
    'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_AVG', 'LANDAREA_AVG', 'LIVINGAREA_AVG', 'FLOORSMAX_MODE',
    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 
    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
    'BURO_BB_MONTHS_BALANCE_SIZE_MEAN', 'PREV_AMT_APPLICATION_MAX',
    'ORGANIZATION_TYPE_XNA', 'WALLSMATERIAL_MODE_Unknown', 'EMERGENCYSTATE_MODE_Unknown',
    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'HOUR_APPR_PROCESS_START', 'DAYS_LAST_PHONE_CHANGE',
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 
    'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 
    'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 
    'NAME_TYPE_SUITE_Unaccompanied', 'NAME_TYPE_SUITE_Unknown', 'NAME_TYPE_SUITE_nan',
    'NAME_INCOME_TYPE_nan', 'NAME_EDUCATION_TYPE_nan', 'NAME_FAMILY_STATUS_nan', 'NAME_HOUSING_TYPE_nan', 'OCCUPATION_TYPE_nan', 
    'ORGANIZATION_TYPE_nan', 'NAME_CONTRACT_TYPE_nan',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
    'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY', 
    'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY',
    'FONDKAPREMONT_MODE_org spec account', 'FONDKAPREMONT_MODE_reg oper account', 
    'FONDKAPREMONT_MODE_reg oper spec account', 'FONDKAPREMONT_MODE_nan',
    'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Monolithic', 'WALLSMATERIAL_MODE_nan', 'WALLSMATERIAL_MODE_Panel', 
    'WALLSMATERIAL_MODE_Stone, brick', 'WALLSMATERIAL_MODE_Wooden', 'HOUSETYPE_MODE_block of flats', 'HOUSETYPE_MODE_specific housing', 
    'HOUSETYPE_MODE_nan', 'EMERGENCYSTATE_MODE_nan','DAYS_BIRTH','DAYS_EMPLOYED', 'BURO_DAYS_CREDIT_MIN','BURO_DAYS_CREDIT_MAX',
    'PREV_AMT_CREDIT_MEAN','PREV_AMT_CREDIT_MEAN', 'PREV_CNT_PAYMENT_MEAN','CC_AMT_BALANCE_MAX','CC_AMT_BALANCE_MIN',
    'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX','WEEKDAY_APPR_PROCESS_START_nan',
]

    
    df_clean = df_clean.drop(columns=[col for col in highly_corr_vars if col in df_clean.columns])

    # 3. Regroupement Organization
    df_clean = regroup_organization_types(df_clean)

    # 4. Regroupement Occupation
    df_clean = regroup_occupation_types(df_clean)
    
    # Si la cible a été retirée au début, on la réajoute ici
    if target_present and target_col:
        df_clean[target_col] = target_values

    return df_clean

