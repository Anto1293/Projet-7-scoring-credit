import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction importée depuis features.py
from api.endpoints import get_all_client_ids, get_client_data
from core.features import  df_all_clients

st.set_page_config(layout="wide")
st.title("📊 Comparaison client vs population")

# Choix du client
client_ids = get_all_client_ids()
selected_id = st.selectbox("Sélectionner un ID client", client_ids)
client_data = get_client_data(selected_id)

# Préparation
df = df_all_clients.copy()
client_row = df[df["SK_ID_CURR"] == selected_id].iloc[0]

# Filtres
st.sidebar.markdown("## 🎯 Filtres comparaison")
age_range = st.sidebar.slider("Âge", int(df["AGE"].min()), int(df["AGE"].max()), (int(df["AGE"].min()), int(df["AGE"].max())))
income_range = st.sidebar.slider("Revenu total", int(df["AMT_INCOME_TOTAL"].min()), int(df["AMT_INCOME_TOTAL"].max()), (int(df["AMT_INCOME_TOTAL"].min()), int(df["AMT_INCOME_TOTAL"].max())))
credit_range = st.sidebar.slider("Montant crédit", int(df["AMT_CREDIT"].min()), int(df["AMT_CREDIT"].max()), (int(df["AMT_CREDIT"].min()), int(df["AMT_CREDIT"].max())))
gender = st.sidebar.multiselect("Genre", options=df["CODE_GENDER"].unique(), default=list(df["CODE_GENDER"].unique()))

# Filtrage
filtered_df = df[
    (df["AGE"] >= age_range[0]) & (df["AGE"] <= age_range[1]) &
    (df["AMT_INCOME_TOTAL"] >= income_range[0]) & (df["AMT_INCOME_TOTAL"] <= income_range[1]) &
    (df["AMT_CREDIT"] >= credit_range[0]) & (df["CODE_GENDER"].isin(gender))
]

st.markdown("### 📌 Comparaison d’une variable")

# Sélection de la feature à comparer
numerical_features = ["AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT"]
selected_feature = st.selectbox("Choisir une variable à comparer", numerical_features)

# Histogramme de distribution
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(filtered_df[selected_feature], kde=False, color="skyblue", bins=30, ax=ax)
ax.axvline(client_row[selected_feature], color="red", linestyle="--", linewidth=2, label="Client sélectionné")
ax.set_title(f"Distribution de {selected_feature}")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Nombre de clients")
ax.legend()
st.pyplot(fig)

st.markdown("### 🔍 Analyse bi-variée")

# Sélection des deux variables pour scatterplot
col1, col2 = st.columns(2)
feature_x = col1.selectbox("Variable X", numerical_features, index=0)
feature_y = col2.selectbox("Variable Y", numerical_features, index=1)

# Scatterplot
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x=feature_x, y=feature_y, alpha=0.5)
ax.scatter(client_row[feature_x], client_row[feature_y], color="red", s=100, label="Client sélectionné")
ax.set_title(f"{feature_y} en fonction de {feature_x}")
ax.legend()
st.pyplot(fig)
