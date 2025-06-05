import streamlit as st
import plotly.graph_objects as go
from core.features import shap_global
from core.features import rename_variable

st.title("🌍 Analyse globale des variables influentes")

# --- Top N features globales ---
top_features_df = shap_global.sort_values("importance", ascending=False).head(8).iloc[::-1]
top_features_df["feature_readable"] = top_features_df["feature"].apply(rename_variable)

# --- Graphique d'importance globale (barres horizontales interactives) ---
fig = go.Figure(go.Bar(
    x=top_features_df["importance"],
    y=top_features_df["feature_readable"],
    orientation='h',
    marker_color='#1f77b4'
))
fig.update_layout(
    title="Top 8 variables les plus influentes (SHAP global)",
    xaxis_title="Importance",
    yaxis_title="Variables",
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
❓️**Interprétation :** Ce graphique affiche les variables qui ont le plus d'influence sur les décisions du modèle pour l'ensemble des clients.
La longueur de la barre reflète l'importance de chaque variable.
""")