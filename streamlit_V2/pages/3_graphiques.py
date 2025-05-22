import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from api.endpoints import get_client_data
from core.features import df_all_clients

# Pour fond sombre graphique
plt.style.use("seaborn-v0_8-colorblind")

st.subheader("Comparaison Client")

# --- Sélection du client ---
client_ids = df_all_clients["SK_ID_CURR"].tolist()
selected_id = st.selectbox("Sélectionner un ID client", client_ids)
client_data = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id].squeeze()

# --- Sélection de la variable ---
feature = st.selectbox("Choisir une variable à comparer", df_all_clients.columns.drop("SK_ID_CURR"))

# --- Comparaison globale ---
col1, col2 = st.columns(2)
# Créer tranches d’âge


with col1:
    st.markdown("#### Répartition globale")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df_all_clients[feature], bins=30, kde=False, ax=ax, color="lightblue")
    ax.axvline(client_data[feature], color="red", linestyle="--", label="Client")
    ax.set_title(f"{feature} - Tous les clients", color= 'black', fontsize=10)
    ax.legend(fontsize=8, labelcolor='grey')
    st.pyplot(fig)

with col2:
    st.markdown("#### Groupe d’âge similaire")
    df_all_clients["AGE_BIN"] = pd.cut(df_all_clients["AGE"], bins=[20, 30, 40, 50, 60, 70, 100], right=False)
    client_age_bin = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id]["AGE_BIN"].values[0]
    group_age = df_all_clients[df_all_clients["AGE_BIN"] == client_age_bin]

    # Construction propre de l'étiquette du groupe d'âge
    age_range = f"[{int(client_age_bin.left)} - {int(client_age_bin.right)})"

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(group_age[feature], bins=20, color="orange", ax=ax2)
    ax2.axvline(client_data[feature], color="red", linestyle="--", label="Client")
    ax2.set_title(f"{feature} - Groupe d’âge {age_range}", color='black', fontsize=10)
    ax2.legend(fontsize=8, labelcolor='grey')
    st.pyplot(fig2)


# --- Scatterplot bi-variable ---
st.markdown("### Analyse bi-variée")

colx, coly = st.columns(2)
with colx:
    x_feature = st.selectbox("Variable X", df_all_clients.columns.drop("SK_ID_CURR"), index=0)
with coly:
    y_feature = st.selectbox("Variable Y", df_all_clients.columns.drop("SK_ID_CURR"), index=1)

fig3, ax3 = plt.subplots(figsize=(4, 2))  # Taille réduite
sns.scatterplot(data=df_all_clients, x=x_feature, y=y_feature, alpha=0.3, color="gray", ax=ax3)
ax3.scatter(client_data[x_feature], client_data[y_feature], color="red", s=40, label="Client")
ax3.set_title(f"{y_feature} vs {x_feature}", fontsize=7, color= 'black')
ax3.tick_params(axis='both', labelsize=7)
ax3.set_xlabel(x_feature, fontsize=7)
ax3.set_ylabel(y_feature, fontsize=7)
ax3.legend(fontsize=6)
st.pyplot(fig3)

