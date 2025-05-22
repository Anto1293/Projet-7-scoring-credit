# app.py

import streamlit as st
from core.features import MENU_TABS, MENU_ICONS, DEFAULT_INDEX

st.set_page_config(page_title="Tableau de bord scoring client", layout="wide")

# Feuille de style accessibilité

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Contenu de la page d'accueil
st.title("🗂️ Bienvenue sur le tableau de bord scoring client")
st.write(
    """
    Guide d'utilisateur :
    - Page 1 : 👤 Interface client (feature importance locale)
    - Page 2 : 🌍 Interface globale (feature importance globale)
    - Page 3 : 📊 Graphiques comparatifs
    - Page 4 : 📋 Interface de simulation
    """
)
