import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from matplotlib import colormaps
import shap
from core.features import shap_global
from api.endpoints import get_all_client_ids, get_client_data, get_prediction
from core.shap_utils import get_client_index, get_shap_explanation, get_top_features
from core.features import rename_variable
import json

st.title("👤 Analyse personnalisée du client")

# --- Sélection client ---
client_ids = get_all_client_ids()
selected_id = st.selectbox("🔍 Sélectionner un ID client", client_ids)
client_data = get_client_data(selected_id)
client_idx = get_client_index(selected_id)
prediction = get_prediction(client_data)

# --- Affichage proba et décision ---
st.metric(label="📉 Probabilité de défaut", value=f"{prediction['proba'] * 100:.2f}%")
st.markdown(f"**Décision du modèle** : `{prediction['décision'].upper()}`")

# --- Jauge de risque (contrastée et interactive) ---
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction['proba'] * 100,
    title={"text": "Score de risque (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#d62728" if prediction['proba'] > 0.1 else "#2ca02c"},
        "steps": [
            {"range": [0, 10], "color": "#98df8a"},
            {"range": [10, 100], "color": "#ff9896"},
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" 
❓️ **Interprétation :** Cette jauge indique le niveau de risque prédit par le modèle. 
Un score supérieur à 10% est considéré comme risqué, le crédit est donc refusé.""")

# --- Top features importantes pour ce client (selon importance globale) ---
st.markdown("### 🧩 Variables les plus influentes pour le client (SHAP global)")
top_features = get_top_features(shap_global, top_n=8)
client_df = pd.DataFrame([client_data]).T.rename(columns={0: "Valeur"})
filtered_df = client_df.loc[top_features]
filtered_df.index = [rename_variable(name) for name in filtered_df.index]
st.dataframe(filtered_df)

st.markdown("""
❓️ **Interprétation :** Ce tableau présente les valeurs des 8 variables les plus importantes globalement pour le client sélectionné.
""")

# --- SHAP local (Waterfall) ---
st.markdown("### 🔍 Explication locale (SHAP waterfall)")

shap_exp = get_shap_explanation(client_idx, client_data)
fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_exp, show=False)
st.pyplot(fig)

st.markdown("""
❓️ **Interprétation :** Ce graphique montre l'impact de chaque variable sur la prédiction du client. 
Les variables à gauche tirent la prédiction vers le défaut, celles à droite augmentent la confiance dans la solvabilité du client.
""")
