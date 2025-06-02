import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from api.endpoints import get_client_data
from core.features import NUMERICAL_FEATURES, BINARY_FEATURES, FEATURE_DESCRIPTIONS, df_all_clients, rename_variable

plt.style.use("seaborn-v0_8-colorblind")

# --- Mapping des valeurs binaires pour affichage lisible ---
BINARY_LABELS = {
    "CODE_GENDER": {0: "Homme", 1: "Femme"},
    "FLAG_OWN_CAR": {0: "Non", 1: "Oui"},
    "FLAG_OWN_REALTY": {0: "Non", 1: "Oui"},
    "NAME_CONTRACT_TYPE_Revolving loans":{0: "Non", 1: "Oui"},
    "NAME_INCOME_TYPE_Commercial associate":{0: "Non", 1: "Oui"},
    "NAME_INCOME_TYPE_Pensioner":{0: "Non", 1: "Oui"},
    "NAME_INCOME_TYPE_State servant":{0: "Non", 1: "Oui"},
    "NAME_INCOME_TYPE_Unemployed":{0: "Non", 1: "Oui"},
    "NAME_INCOME_TYPE_Working":{0: "Non", 1: "Oui"},
    "NAME_EDUCATION_TYPE_Higher education":{0: "Non", 1: "Oui"},
    "NAME_EDUCATION_TYPE_Lower secondary":{0: "Non", 1: "Oui"},
    "NAME_EDUCATION_TYPE_Secondary / secondary special":{0: "Non", 1: "Oui"},
    "NAME_FAMILY_STATUS_Married":{0: "Non", 1: "Oui"},
    "NAME_FAMILY_STATUS_Single / not married":{0: "Non", 1: "Oui"},
    "NAME_FAMILY_STATUS_Widow":{0: "Non", 1: "Oui"},
    "NAME_HOUSING_TYPE_House / apartment":{0: "Non", 1: "Oui"},
    "NAME_HOUSING_TYPE_Office apartment":{0: "Non", 1: "Oui"},
    "NAME_HOUSING_TYPE_Rented apartment":{0: "Non", 1: "Oui"},
    "NAME_HOUSING_TYPE_With parents":{0: "Non", 1: "Oui"},
    "SECTOR_Industry":{0: "Non", 1: "Oui"},
    "SECTOR_Trade":{0: "Non", 1: "Oui"},
    "SECTOR_Transport":{0: "Non", 1: "Oui"},
    "SECTOR_Business Entity":{0: "Non", 1: "Oui"},
    "SECTOR_Government":{0: "Non", 1: "Oui"},
    "SECTOR_Security":{0: "Non", 1: "Oui"},
    "SECTOR_Services":{0: "Non", 1: "Oui"},
    "SECTOR_Construction":{0: "Non", 1: "Oui"},
    "SECTOR_Medicine":{0: "Non", 1: "Oui"},
    "SECTOR_Police":{0: "Non", 1: "Oui"},
    "SECTOR_Other":{0: "Non", 1: "Oui"},
    "OCCUPATION_Labor_Work":{0: "Non", 1: "Oui"},
    "OCCUPATION_Sales_Services":{0: "Non", 1: "Oui"},
    "OCCUPATION_Medical_Staff":{0: "Non", 1: "Oui"},
    "OCCUPATION_Security":{0: "Non", 1: "Oui"},
    "OCCUPATION_Management_Core":{0: "Non", 1: "Oui"},
    "OCCUPATION_Other":{0: "Non", 1: "Oui"}}

def apply_labels(df, feature):
    """Applique les labels lisibles aux colonnes binaires si connues."""
    if feature in BINARY_LABELS:
        return df[feature].map(BINARY_LABELS[feature])
    return df[feature]

st.subheader("📊 Comparaison Client")

# --- Sélection du client ---
client_ids = df_all_clients["SK_ID_CURR"].tolist()
selected_id = st.selectbox("Sélectionner un ID client", client_ids)
client_data = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id].squeeze()

# --- Variable à comparer ---
all_features = df_all_clients.columns.drop("SK_ID_CURR")
feature = st.selectbox("📌 Choisir une variable à comparer", all_features, format_func=rename_variable)

# --- Comparaison simple : tous les clients & groupe d'âge ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Répartition dans la population")
    fig, ax = plt.subplots(figsize=(5, 3))
    display_feature = apply_labels(df_all_clients, feature)
    sns.histplot(display_feature, bins=30, kde=False, ax=ax, color="lightblue")
    client_val = apply_labels(client_data.to_frame().T, feature).values[0]
    ax.axvline(client_val, color="red", linestyle="--", label="Client")
    ax.set_title(f"{rename_variable(feature)} - Tous les clients", fontsize=10)
    ax.set_xlabel(rename_variable(feature))
    ax.legend(fontsize=8)
    st.pyplot(fig)

with col2:
    st.markdown("#### Comparaison dans votre groupe d’âge")
    df_all_clients["AGE_BIN"] = pd.cut(df_all_clients["AGE"], bins=[20, 30, 40, 50, 60, 70, 100], right=False)
    client_age_bin = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id]["AGE_BIN"].values[0]
    group_age = df_all_clients[df_all_clients["AGE_BIN"] == client_age_bin]
    age_range = f"[{int(client_age_bin.left)} - {int(client_age_bin.right)})"

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    display_feature_age = apply_labels(group_age, feature)
    sns.histplot(display_feature_age, bins=20, color="orange", ax=ax2)
    client_val_age = apply_labels(client_data.to_frame().T, feature).values[0]
    ax2.axvline(client_val_age, color="red", linestyle="--", label="Client")
    ax2.set_title(f"{rename_variable(feature)} - Groupe d’âge {age_range}", fontsize=10)
    ax2.set_xlabel(rename_variable(feature))
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

# --- Analyse bi-variée ---
st.markdown("### 🔍 Analyse croisée entre deux variables")

# Sélection des variables
colx, coly = st.columns(2)
with colx:
    x_feature = st.selectbox("Variable X", all_features, format_func=rename_variable, index=0)
with coly:
    y_feature = st.selectbox("Variable Y", all_features, format_func=rename_variable, index=1)

# Détection des types
is_x_num = x_feature in NUMERICAL_FEATURES
is_y_num = y_feature in NUMERICAL_FEATURES

fig3, ax3 = plt.subplots(figsize=(5, 3))

if is_x_num and is_y_num:
    # Cas Quantitatif vs Quantitatif
    sns.regplot(data=df_all_clients, x=x_feature, y=y_feature, scatter=False, ax=ax3, color="gray")
    ax3.scatter(client_data[x_feature], client_data[y_feature], color="red", label="Client", s=40)
    ax3.set_title(f"{rename_variable(y_feature)} en fonction de {rename_variable(x_feature)}")
    ax3.set_xlabel(rename_variable(x_feature))
    ax3.set_ylabel(rename_variable(y_feature))
    st.pyplot(fig3)

elif not is_x_num and is_y_num:
    # Cas Catégoriel vs Quantitatif
    x_vals = apply_labels(df_all_clients, x_feature)
    sns.barplot(x=x_vals, y=df_all_clients[y_feature], estimator=np.mean, ax=ax3, palette="Blues_d")
    ax3.scatter(
        [apply_labels(client_data.to_frame().T, x_feature).values[0]],
        [client_data[y_feature]],
        color="red",
        label="Client"
    )
    ax3.set_title(f"Moyenne de {rename_variable(y_feature)} par {rename_variable(x_feature)}")
    ax3.set_xlabel(rename_variable(x_feature))
    ax3.set_ylabel(rename_variable(y_feature))
    st.pyplot(fig3)

elif is_x_num and not is_y_num:
    # Cas Quantitatif vs Catégoriel
    y_vals = apply_labels(df_all_clients, y_feature)
    sns.barplot(y=y_vals, x=df_all_clients[x_feature], estimator=np.mean, ax=ax3, palette="Blues_d", orient="h")
    ax3.scatter(
        [client_data[x_feature]],
        [apply_labels(client_data.to_frame().T, y_feature).values[0]],
        color="red",
        label="Client"
    )
    ax3.set_title(f"Moyenne de {rename_variable(x_feature)} par {rename_variable(y_feature)}")
    ax3.set_ylabel(rename_variable(y_feature))
    ax3.set_xlabel(rename_variable(x_feature))
    st.pyplot(fig3)

elif not is_x_num and not is_y_num:
    # Cas Catégoriel vs Catégoriel
    x_vals = apply_labels(df_all_clients, x_feature)
    y_vals = apply_labels(df_all_clients, y_feature)

    # Créer une table de contingence normalisée ligne (pourcentages par x_feature)
    ct = pd.crosstab(x_vals, y_vals, normalize='index') * 100

    # Plot
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bars = ct.plot(kind="bar", stacked=True, colormap="Set2", ax=ax3)

    # Ajout des étiquettes de pourcentage dans les barres
    for container in bars.containers:
        bars.bar_label(container, fmt="%.0f%%", label_type="center", fontsize=8)

    # Titre et axes
    ax3.set_title(
        f"Répartition (%) de {rename_variable(y_feature)} selon {rename_variable(x_feature)}",
        fontsize=11
    )
    ax3.set_xlabel(rename_variable(x_feature))
    ax3.set_ylabel("Pourcentage (%)")
    ax3.legend(title=rename_variable(y_feature), fontsize=8, title_fontsize=9)
    st.pyplot(fig3)

else:
    st.warning("Ce type de croisement n’est pas encore géré.")
