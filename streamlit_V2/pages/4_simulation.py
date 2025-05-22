# pages/4_Simulation.py

import streamlit as st
import pandas as pd
from core.inputs import display_grouped_inputs, build_data_for_api
from api.endpoints import get_prediction, get_client_data, get_all_client_ids
from core.predictions_utils import display_one_hot_selectbox

st.title("🧪 Simulation de Scénarios de Crédit")

# Sélection de l'ID client
client_ids = get_all_client_ids()
selected_id = st.selectbox("🔍 Sélectionner un ID client :", client_ids)

# Récupération de la ligne du client
client_data = get_client_data(selected_id)
# Conversion en DataFrame
original_data = pd.Series(client_data)

st.markdown("#### ✏️ Modifier les caractéristiques du client")
modified_inputs = display_grouped_inputs(original_data)

# Construction de la ligne de données à envoyer à l'API
data_for_api = build_data_for_api(modified_inputs, {}, original_data)

# Bouton de prédiction
if st.button("🔍 Prédire la probabilité de défaut"):
    try:
        result = get_prediction(data_for_api)
        proba = result.get("proba", 0)
        prediction = 1 if proba >= 0.10 else 0

        st.markdown(f"### 🔢 Probabilité de défaut : **{proba:.2%}**")
        if prediction:
            st.error("❌ Crédit REFUSÉ (risque trop élevé)")
        else:
            st.success("✅ Crédit ACCEPTÉ (risque acceptable)")
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")
