import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from matplotlib import colormaps
import shap
from core.features import shap_global
from api.endpoints import get_all_client_ids, get_client_data, get_prediction
from core.shap_utils import get_client_index, get_shap_explanation, get_top_features

st.subheader("Vue Client")

# --- Sélection client ---
client_ids = get_all_client_ids()
selected_id = st.selectbox("Sélectionner un ID client", client_ids)
client_data = get_client_data(selected_id)
client_idx = get_client_index(selected_id)
prediction = get_prediction(client_data)

# --- Affichage proba et décision ---
st.metric(label="Probabilité de défaut", value=f"{prediction['proba'] * 100:.2f}%")
st.markdown(f"**Décision du modèle** : `{prediction['décision'].upper()}`")

# --- Jauge de risque ---
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction['proba'] * 100,
    title={"text": "Score de risque (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red" if prediction['proba'] > 0.1 else "green"},
        "steps": [
            {"range": [0, 10], "color": "lightgreen"},
            {"range": [10, 100], "color": "lightcoral"},
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)

# --- Top features importantes pour ce client (selon importance globale) ---
top_features = get_top_features(shap_global, top_n=8)
client_df = pd.DataFrame([client_data]).T.rename(columns={0: "Valeur"})
filtered_df = client_df.loc[top_features]
st.markdown("### Variables clés du client")
st.dataframe(filtered_df)

# Palette daltonien-friendly
color_positive = "#E69F00"  # orange
color_negative = "#56B4E9"  # bleu clair

# --- SHAP local (waterfall) ---
st.markdown("### Explication SHAP locale")

# Explication SHAP du client
shap_exp = get_shap_explanation(client_idx, client_data)
fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_exp, show=False)
st.pyplot(fig)
