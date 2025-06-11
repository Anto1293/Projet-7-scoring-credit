import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import textwrap
from sklearn.linear_model import LinearRegression
from api.endpoints import get_client_data
from core.features import NUMERICAL_FEATURES, BINARY_FEATURES, df_all_clients, rename_variable

client_color = '#FF4136'  # rouge vif

# Section : Comparaison Client
st.title("📊 Comparaison client avec la population")

st.markdown("""
Cette section permet de comparer les caractéristiques du client sélectionné par rapport à l'ensemble de la population,
ainsi qu'à son groupe d'âge. Les graphiques sont interactifs pour une meilleure exploration.
""")

# Sélection du client et de la variable à comparer
client_ids = df_all_clients["SK_ID_CURR"].tolist()   # Liste des identifiants clients
selected_id = st.selectbox("Sélectionner un ID client", client_ids) # Menu déroulant 
client_data = df_all_clients.loc[df_all_clients["SK_ID_CURR"] == selected_id].iloc[0]
# Extrait les données du client sélectionné (sous forme de série)

all_features = df_all_clients.columns.drop("SK_ID_CURR")
feature = st.selectbox("Choisir une variable à comparer", all_features, format_func=rename_variable) # Menu déroulant avec noms formatés

# --- Mapping des valeurs binaires pour affichage lisible (# Dictionnaire qui mappe les valeurs binaires (0/1) vers des labels lisibles ("Oui"/"Non", "Homme"/"Femme"))
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
    if feature in BINARY_LABELS:
        return df[feature].map(BINARY_LABELS[feature])
    return df[feature]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Répartition dans la population")
    st.caption("❓️Ce graphique montre la distribution de la variable choisie dans l'ensemble des clients. La ligne rouge indique la position du client sélectionné.")
    x_values = apply_labels(df_all_clients, feature)
    client_val = apply_labels(client_data.to_frame().T, feature).values[0]

    fig = px.histogram(x=x_values, nbins=20, title=rename_variable(feature), labels={"x": rename_variable(feature)}, color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.add_vline(x=client_val, line_dash="dash", line_color="red")
    fig.update_layout(title_text=f"{rename_variable(feature)} - Tous les clients", 
                      font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    st.markdown("#### Comparaison dans votre groupe d’âge")
    st.caption("❓️Ce graphique montre la même variable, mais uniquement pour le groupe d'âge du client. Cela permet de faire une comparaison plus ciblée.")
    
    df_all_clients["AGE_BIN"] = pd.cut(df_all_clients["AGE"], bins=[20, 30, 40, 50, 60, 70, 100], right=False)
    client_age_bin = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id]["AGE_BIN"].values[0]
    age_group = df_all_clients[df_all_clients["AGE_BIN"] == client_age_bin]

    x_values_age_group = apply_labels(age_group, feature)

    fig2 = px.histogram(x=x_values_age_group, nbins=20, color_discrete_sequence=px.colors.qualitative.Dark2,
                        title=f"{rename_variable(feature)} – Groupe {int(client_age_bin.left)}-{int(client_age_bin.right)} ans",
                        labels={"x": rename_variable(feature)})
    fig2.add_vline(x=client_val, line_dash="dash", line_color="red")
    fig2.update_layout( font=dict(color='white'))
    st.plotly_chart(fig2, use_container_width=True)

# --- Partie 2 : Analyse croisée ---
st.title("🔍 Analyse croisée entre deux variables")
st.markdown("""
Cette section explore les relations entre deux variables. Les graphiques changent dynamiquement selon la nature des variables sélectionnées.
""")

colx, coly = st.columns(2)
with colx:
    x_feature = st.selectbox("Variable X", all_features, format_func=rename_variable, index=0)
with coly:
    y_feature = st.selectbox("Variable Y", all_features, format_func=rename_variable, index=1)

is_x_num = x_feature in NUMERICAL_FEATURES
is_y_num = y_feature in NUMERICAL_FEATURES

st.caption("Le type de graphique est automatiquement adapté au type des variables sélectionnées.")
fig3 = go.Figure()

# 1. Deux variables quantitatives
if is_x_num and is_y_num:
    # Droite de régression uniquement
    x = df_all_clients[x_feature].values.reshape(-1, 1)   # Récupère les valeurs X en 2D (obligatoire pour sklearn)
    y = df_all_clients[y_feature].values                  # Les valeurs Y correspondantes
    model = LinearRegression().fit(x, y)                  # entraîne un modèle de régression linéaire simple pour prédire y à partir de x
    y_pred = model.predict(x)                             # calcule les valeurs prédites (droite de régression) sur tous les points x


    fig3.add_trace(go.Scatter(x=df_all_clients[x_feature], y=y_pred, mode='lines',
                              name='Régression', line=dict(color="orange")))
    fig3.add_trace(go.Scatter(x=[client_data[x_feature]], y=[client_data[y_feature]], mode='markers',
                              name='Client', marker=dict(color='red', size=12, line=dict(color='white', width=1.5))))

    fig3.update_layout(title=f"Régression : {rename_variable(y_feature)} selon {rename_variable(x_feature)}",
                       xaxis_title=rename_variable(x_feature),
                       yaxis_title=rename_variable(y_feature),
                       xaxis=dict(showgrid=True, gridcolor='gray'),
                       yaxis=dict(showgrid=True, gridcolor='gray'),
                       font=dict(color='white'))
    fig3.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)


elif not is_x_num and is_y_num:
    x_vals = apply_labels(df_all_clients, x_feature)
    df_plot = df_all_clients.copy()
    df_plot["x_vals"] = x_vals

    # Calcul de la moyenne de y_feature par catégorie x_feature
    df_mean = df_plot.groupby("x_vals")[y_feature].mean().reset_index()

    fig3 = px.bar(
        df_mean,
        x="x_vals",
        y=y_feature,
        color="x_vals",
        color_discrete_sequence=px.colors.qualitative.Dark2,
        labels={"x_vals": rename_variable(x_feature), y_feature: rename_variable(y_feature)},
        title=f"Moyenne de {rename_variable(y_feature)} par {rename_variable(x_feature)}"
    )

    # Point du client
    client_x_val = str(apply_labels(client_data.to_frame().T, x_feature).values[0])
    fig3.add_trace(go.Scatter(
        x=[client_x_val], y=[client_data[y_feature]], mode='markers',
        name='Client', marker=dict(color=client_color, size=14, line=dict(color='white', width=1.5))
    ))


elif is_x_num and not is_y_num:
    y_vals = apply_labels(df_all_clients, y_feature)
    df_plot = df_all_clients.copy()
    df_plot["y_vals"] = y_vals

    # Calcul de la moyenne de x_feature par catégorie y_feature
    df_mean = df_plot.groupby("y_vals")[x_feature].mean().reset_index()

    fig3 = px.bar(
        df_mean,
        y="y_vals",
        x=x_feature,
        color="y_vals",
        color_discrete_sequence=px.colors.qualitative.Dark2,
        orientation='h',
        labels={"y_vals": rename_variable(y_feature), x_feature: rename_variable(x_feature)},
        title=f"Moyenne de {rename_variable(x_feature)} par {rename_variable(y_feature)}"
    )

    # Point du client
    client_y_val = str(apply_labels(client_data.to_frame().T, y_feature).values[0])
    fig3.add_trace(go.Scatter(
        y=[client_y_val], x=[client_data[x_feature]], mode='markers',
        name='Client', marker=dict(color=client_color, size=14, line=dict(color='white', width=1.5))
    ))


elif not is_x_num and not is_y_num:
    x_vals = apply_labels(df_all_clients, x_feature)
    y_vals = apply_labels(df_all_clients, y_feature)

    ct = pd.crosstab(x_vals, y_vals, normalize='index') * 100
    ct.index.name = rename_variable(x_feature)
    ct.columns.name = rename_variable(y_feature)


    fig3 = px.bar(
        ct, barmode='stack',
        title=f"Répartition (%) de {rename_variable(y_feature)} selon {rename_variable(x_feature)}",
        color_discrete_sequence=px.colors.qualitative.Dark2,
        labels={'value': 'Pourcentage (%)', 'index': rename_variable(x_feature)},
    )

    client_x_val = str(apply_labels(client_data.to_frame().T, x_feature).values[0])
    client_y_val = str(apply_labels(client_data.to_frame().T, y_feature).values[0])

    # Le client est représenté par un point rouge sur le bar empilé
    fig3.add_trace(go.Scatter(
        x=[client_x_val],
        y=[ct.loc[client_x_val, client_y_val]],
        mode="markers+text",
        marker=dict(size=12, color="red", symbol="circle"),
        text=["Client"],
        textposition="top center",
        name="Client",
        line=dict(color='white', width=1.5)
))

    fig3.update_xaxes(autorange=True)
    fig3.update_yaxes(autorange=True)
    

else:
    st.warning("Ce type de croisement n’est pas encore géré.")

st.plotly_chart(fig3, use_container_width=True)


st.markdown("""
### ❓️ Interprétation des graphiques croisés

Ce graphique interactif montre la relation entre les deux variables sélectionnées.  
Il s’adapte automatiquement selon le type de variables (quantitatives ou catégorielles).

- 📈 **Deux variables quantitatives** :  
  Une **droite de régression** est affichée pour visualiser la tendance générale.  
  Une pente positive indique que lorsque la variable X augmente, la variable Y augmente également (et inversement pour une pente négative).

- 📊 **Une variable catégorielle et une variable quantitative** :  
  Un **graphique en barres** affiche la **moyenne** de la variable quantitative pour chaque groupe.  
  Cela permet de comparer visuellement les différentes catégories (ex : revenu moyen selon le statut familial).

- 🧩 **Deux variables catégorielles** :  
  Un **graphique en barres** affiche la **répartition en pourcentage** de la variable Y pour chaque modalité de la variable X.  
  Cela permet d’identifier les groupes les plus représentés ou les plus équilibrés.

🔎 **La position du client est indiquée sur chaque graphique** par un repère visuel (point rouge), permettant de le situer dans la population.
""")

