# features.py

import pandas as pd

# Chargement des ressources distantes
DATA_LOCAL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_values_clients.parquet"
DATA_BASE = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_base_value.parquet"
DATA_GLOBAL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_global_importance"
DATA_ALL_CLIENTS = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/train_subset_with_target.parquet"
IDS_URL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/ids_clients.csv"

shap_values = pd.read_parquet(DATA_LOCAL)  # valeurs des shap locales par clients
base_value = pd.read_parquet(DATA_BASE).values[0]    # Valeur de base SHAP
shap_global = pd.read_parquet(DATA_GLOBAL) # feature importance globale
ids_clients = pd.read_csv(IDS_URL)         # Liste des IDs des clients
df_all_clients = pd.read_parquet(DATA_ALL_CLIENTS) # pour les graphiques comparatifs

MENU_TABS = ["Accueil", "Vue client", "Vue globale", "Comparaison", "Simulation"]
MENU_ICONS = ["🗂️", "🧍", "🌍", "📊", "📋"]
DEFAULT_INDEX = 0

# Définition des features pour simulation

NUMERICAL_FEATURES = ["CNT_CHILDREN", "AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE","INCOME_PER_PERSON",
    "ANNUITY_INCOME_PERC", "PAYMENT_RATE", "CREDIT_TERM", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "YEARS_BUILD_AVG", "TOTALAREA_MODE", "BURO_DAYS_CREDIT_MEAN", "BURO_AMT_CREDIT_SUM_MEAN",
    "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN", "BURO_CREDIT_DAY_OVERDUE_MEAN", "PREV_AMT_CREDIT_MAX", "PREV_AMT_APPLICATION_MEAN", "PREV_APP_CREDIT_PERC_MEAN",
    "PREV_DAYS_DECISION_MIN", "POS_SK_DPD_MEAN", "POS_SK_DPD_DEF_MEAN", "INSTAL_PAYMENT_DIFF_MEAN", "INSTAL_PAYMENT_DIFF_MAX",
    "INSTAL_DPD_MEAN", "CC_AMT_BALANCE_MEAN", "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN", "CC_SK_DPD_MAX", "DAYS_EMPLOYED_PERC"]

BINARY_FEATURES = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

OHE_FEATURES = [
    "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION", "SECTOR"
]

# Mapping pour retrouver les préfixes OHE (ex: "NAME_FAMILY_STATUS_" etc.)
OHE_MAPPING = {
    "NAME_CONTRACT_TYPE": "NAME_CONTRACT_TYPE_",
    "NAME_INCOME_TYPE": "NAME_INCOME_TYPE_",
    "NAME_EDUCATION_TYPE": "NAME_EDUCATION_TYPE_",
    "NAME_FAMILY_STATUS": "NAME_FAMILY_STATUS_",
    "NAME_HOUSING_TYPE": "NAME_HOUSING_TYPE_",
    "OCCUPATION": "OCCUPATION_",
    "SECTOR": "SECTOR_"
}

# Mapping pour onglet simulation

FEATURE_GROUPS = {
    "📋 Informations personnelles": ["CODE_GENDER", "CNT_CHILDREN", "AGE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE"],
    "🚗 Possessions": ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"],
    "💼 Emploi & revenus": ["AMT_INCOME_TOTAL", "INCOME_PER_PERSON", "NAME_INCOME_TYPE", "OCCUPATION", "SECTOR", "DAYS_EMPLOYED_PERC"],
    "🏠 Région & Logement": ["REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT_W_CITY", "YEARS_BUILD_AVG", "TOTALAREA_MODE"],
    "💳 Crédit et historiques de crédit": ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE", "ANNUITY_INCOME_PERC", "PAYMENT_RATE", "CREDIT_TERM",
                                         "BURO_DAYS_CREDIT_MEAN", "BURO_AMT_CREDIT_SUM_MEAN", "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN", "BURO_CREDIT_DAY_OVERDUE_MEAN", "PREV_AMT_CREDIT_MAX",
                                          "PREV_AMT_APPLICATION_MEAN", "PREV_APP_CREDIT_PERC_MEAN", "PREV_DAYS_DECISION_MIN", "POS_SK_DPD_MEAN", "POS_SK_DPD_DEF_MEAN", 
                                          "INSTAL_PAYMENT_DIFF_MEAN", "INSTAL_PAYMENT_DIFF_MAX","INSTAL_DPD_MEAN"],
    "🔍 Sources externes": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "📆 Informations bancaires": ["CC_AMT_BALANCE_MEAN", "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN", "CC_SK_DPD_MAX"]
}


# Ce dictionnaire contient les info-bulles par feature pour enrichir l'affichage
FEATURE_DESCRIPTIONS = {
    "REGION_POPULATION_RELATIVE": "→ Proportion de la population vivant dans la région du client par rapport à la population totale. Valeurs proches de 0 = région peu peuplée, proches de 1 = très peuplée.",
    "REGION_RATING_CLIENT_W_CITY": "→ Évaluation (1 à 3) de la qualité de la région où vit le client (1 = mauvaise, 3 = bonne).",
    "EXT_SOURCE_1": " → Scores de risque externes (fournis par des sources tierces). Plus le score est élevé (proche de 1), plus le client est considéré comme fiable.",
    "YEARS_BUILD_AVG": "→ Année moyenne de construction des bâtiments de la zone, normalisée. Valeur proche de 0 = anciens bâtiments, proche de 1 = récents.",
    "TOTALAREA_MODE": "→ Représente la surface totale du logement du client, normalisée entre 0 et 1. Par exemple : 0.1 = petit logement, 0.5 = moyen, 0.9 = grand logement.",
    "BURO_DAYS_CREDIT_MEAN": "→ Ancienneté moyenne des crédits passés (en jours). Valeurs négatives (ex. -1500) indiquent l’ancienneté par rapport à aujourd’hui",
    "BURO_AMT_CREDIT_SUM_MEAN": "→ Montant moyen des crédits passés enregistrés dans les bureaux de crédit.",
    "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN": "→ Montant moyen des crédits en retard (non remboursés à temps).", 
    "BURO_CREDIT_DAY_OVERDUE_MEAN": "→ Durée moyenne des retards de paiement (en jours).", 
    "PREV_AMT_CREDIT_MAX": " → Montant maximal accordé sur un crédit précédent.", 
    "PREV_AMT_APPLICATION_MEAN": "→ Montant moyen demandé sur des crédits précédents",
    "PREV_APP_CREDIT_PERC_MEAN": "→ Ratio entre montant du crédit accordé et montant demandé",
    "PREV_DAYS_DECISION_MIN": "→ Date la plus ancienne (en jours) d’une décision de crédit passée",
    "POS_SK_DPD_MEAN": "→ Nombre moyen de jours de retard (DPD) sur les crédits en cours.", 
    "POS_SK_DPD_DEF_MEAN": "→ Nombre moyen de jours de retard sérieux (défauts) sur POS.", 
    "INSTAL_PAYMENT_DIFF_MEAN": "→ Ecart moyen entre paiement prévu et paiement réel.", 
    "INSTAL_PAYMENT_DIFF_MAX": "→ Ecart maximum observé entre paiement prévu et réalisé.", 
    "INSTAL_DPD_MEAN": "→ Jours de retard moyen sur les échéances d’emprunt.", 
    "CC_AMT_BALANCE_MEAN": "→ Solde moyen des cartes de crédit.", 
    "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN": "→ Montant moyen des paiements mensuels sur les cartes.",
    "CC_SK_DPD_MAX": "→ Nombre maximum de jours de retard sur les cartes de crédit.",
    "DAYS_EMPLOYED_PERC": "→ Ratio entre le nombre de jours travaillés et l'âge du client. Permet d’estimer la stabilité professionnelle par rapport à l’âge (valeurs proches de 1 = long historique d’emploi)."}
