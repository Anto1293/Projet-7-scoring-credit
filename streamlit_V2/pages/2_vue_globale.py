import streamlit as st
import plotly.graph_objects as go
from core.features import shap_global

st.subheader("Vue Globale")

# --- Top N features globales ---
top_features_df = shap_global.sort_values("importance", ascending=False).head(8).iloc[::-1] #.iloc pour inverser le tableau


# --- Graphique d'importance globale ---
fig = go.Figure(go.Bar(
    x=top_features_df["importance"],
    y=top_features_df["feature"],
    orientation='h',
    marker_color='darkblue'
))
fig.update_layout(
    title="Top 8 variables (importance SHAP globale)",
    xaxis_title="Importance",
    yaxis_title="Variable",
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig, use_container_width=True)
