import pandas as pd
import shap
from core.features import shap_values, base_value, shap_global, ids_clients, shap_global

# ----- Fonctions Streamlit -----
def get_client_index(client_id): 
    return ids_clients[ids_clients["SK_ID_CURR"] == client_id].index[0] # Renvoie l’index d’un client à partir de son ID

def get_shap_explanation(client_idx, client_data):
    # Convertir client_data dict en DataFrame ligne pour récupérer les features correctement
    data_df = pd.DataFrame([client_data]).drop(columns=["SK_ID_CURR"], errors='ignore') # Transforme les données client en DataFrame (1 ligne)
    return shap.Explanation(
        values=shap_values.iloc[client_idx].values, # Valeurs SHAP du client
        base_values=base_value,  # Valeur de base SHAP
        data=data_df.iloc[0].values, # Données d'entrée du client
        feature_names=data_df.columns.tolist()) # Noms des variables

def get_top_features(shap_global_df, top_n=8):
    """Retourne les noms des N features les plus importantes selon SHAP global"""
    return shap_global_df.sort_values("importance", ascending=False)["feature"].head(top_n).tolist()