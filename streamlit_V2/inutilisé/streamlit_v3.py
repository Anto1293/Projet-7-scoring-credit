import requests
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#-----API endpoint-----
API_URL = "https://projet-7-scoring-credit.onrender.com"

# Chargement pour SHAP
DATA_LOCAL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_values_clients.parquet"
DATA_BASE = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_base_value.parquet"
DATA_GLOBAL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/shap_global_importance"
IDS_URL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/ids_clients.csv"
DATA_ALL_CLIENTS = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/train_subset_with_target.parquet"

# ----- Chargement des données -----
shap_values = pd.read_parquet(DATA_LOCAL)  # valeurs des shap locales par clients
base_value = pd.read_parquet(DATA_BASE).values[0]    # Valeur de base SHAP
shap_global = pd.read_parquet(DATA_GLOBAL) # feature importance globale
ids_clients = pd.read_csv(IDS_URL)         # Liste des IDs des clients
df_all_clients = pd.read_parquet(DATA_ALL_CLIENTS) # pour les graphiques comparatifs

# ----- Fonctions API -----
def get_all_client_ids():
    return requests.get(f"{API_URL}/client/").json() # Récupère tous les IDs clients depuis l'API 

def get_client_data(client_id):
    return requests.get(f"{API_URL}/client/{client_id}").json()  # Récupère les données d’un client donné via son ID

def get_prediction(client_data):
    return requests.post(f"{API_URL}/predict", json=client_data).json() # Envoie les données du client à l’API pour prédiction

# ----- Fonctions Streamlit -----
def get_client_index(client_id): 
    return ids_clients[ids_clients["SK_ID_CURR"] == client_id].index[0] # Renvoie l’index d’un client à partir de son ID

def get_shap_explanation(client_idx, client_data):
    # Convertir client_data dict en DataFrame ligne pour récupérer les features correctement
    data_df = pd.DataFrame([client_data]).drop(columns=["SK_ID_CURR"], errors='ignore') # Transforme les données client en DataFrame (1 ligne)
    return shap.Explanation(
        values=shap_values.iloc[client_idx].values, # Valeurs SHAP du client
        base_values=base_value,  # Valeur de base SHAP
        data=data_df.iloc[0].values, # Données d'entrée du client
        feature_names=data_df.columns.tolist()) # Noms des variables

# Fonction pour récupérer les valeurs OHE
def display_one_hot_selectbox(data_row, column_prefix, label):
    """
    Affiche un selectbox Streamlit à partir de colonnes one-hot encodées.
    Args:
        data_row (pd.Series): Une ligne du DataFrame
        column_prefix (str): Préfixe des colonnes (ex: "NAME_FAMILY_STATUS_")
        label (str): Libellé du selectbox
    Returns:
        str: Valeur sélectionnée
    """
    # Sélectionne les colonnes correspondant au préfixe
    matching_columns = [col for col in data_row.index if col.startswith(column_prefix)]
    # Récupère les options à afficher (valeurs originales)
    options = [col.replace(column_prefix, "") for col in matching_columns]
    # Cherche l’option sélectionnée (où la valeur est 1)
    default_col = next((col for col in matching_columns if data_row[col] == 1), None)
    # Valeur par défaut sélectionnée dans la liste
    default_value = default_col.replace(column_prefix, "") if default_col else options[0]
    # Afficher le selectbox
    return st.selectbox(label, options=options, index=options.index(default_value))

# ----- Configuration Streamlit -----
st.set_page_config(page_title="Scoring Client", layout="wide")
st.title("Tableau de bord Scoring Client - Version 2")

# ----- Menu latéral -----
with st.sidebar:
    st.header("Navigation")
    selected_menu = option_menu("Onglets",["Vue client", "Vue globale", "Comparaison", "Simulation"], # Liste des onglets
        icons=["person", "globe", "bar-chart", "sliders"], # Icônes correspondantes pour chaque onglet
        menu_icon="cast", # Icône pour le menu
        default_index=0) # Onglet par défaut sélectionné (index 0)

# ----- Vue Client -----
if selected_menu == "Vue client":
    st.subheader("Analyse locale du client")

    client_ids = get_all_client_ids() # Récupère tous les IDs clients depuis l’API
    selected_id = st.selectbox("Sélectionner un ID client", client_ids)  # Sélectbox d’ID client dans Streamlit
    client_data = get_client_data(selected_id)   # Données du client sélectionné (API) (dict)
    client_idx = get_client_index(selected_id)   # Index du client (pour shap_values)
    prediction = get_prediction(client_data)    # Récupère la prédiction depuis l’API
    original_data = pd.DataFrame([client_data])

    # Affichage décision
    st.metric(label="Probabilité de défaut", value=f"{prediction['proba']*100:.2f}%")
    st.markdown(f"**Décision du modèle** : {prediction['décision'].upper()}")  # Affiche la décision du modèle en MAJ

    # Jauge de risque: go=(plotly graph object = module de Plotly pour créer des visualisations)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction['proba']*100,
        title={"text": "Score de risque (%)"},
        gauge={
            "axis": {"range": [0, 100]}, # Axe entre 0 et 100
            "bar": {"color": "red" if prediction['proba'] > 0.1 else "green"},
            "steps": [
                {"range": [0, 10], "color": "lightgreen"},
                {"range": [10, 100], "color": "lightcoral"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)  # Affiche le graphique dans Streamlit

    # Données client (st.expander =  section pliable / dépliable)
    with st.expander("Données du client"):
         st.dataframe(pd.DataFrame([client_data]).T.rename(columns={0: "Valeur"})) 
        # Affiche les données en transposant les lignes (pour plus de visibilité)
        # Renomme la seule colonne existante (0) en "Valeur" pour plus de clarté

    # SHAP local
    with st.expander("Explication SHAP du client"):
        shap_exp = get_shap_explanation(client_idx, client_data) # Récupère l’objet SHAP
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, show=False) # Graphique waterfall SHAP
        st.pyplot(fig)

# ----- Vue Globale -----
elif selected_menu == "Vue globale":
    st.subheader("Importance globale des variables")
    # Affiche le tableau des variables triées par importance décroissante
    st.dataframe(shap_global.sort_values("importance", ascending=False))
    # Crée une figure Plotly : graphique en barres horizontales des importances SHAP
    fig = go.Figure(go.Bar(
        x=shap_global.sort_values("importance", ascending=True)["importance"],  # Valeurs triées dans l'ordre croissant
        y=shap_global.sort_values("importance", ascending=True)["feature"],     # Noms des variables correspondantes 
        orientation='h',  # Orientation horizontale
        marker_color='darkblue'
    ))
    # Mise en forme du graphique
    fig.update_layout(
        title="Top variables par importance SHAP",
        xaxis_title="Importance",
        yaxis_title="Variable", 
        margin=dict(l=0, r=0, t=40, b=0)  # marges
    )
    # Affiche le graphique
    st.plotly_chart(fig, use_container_width=True)

# ----- Comparaison -----
elif selected_menu == "Comparaison":
    st.subheader("Comparaison client / population")



# ----- Simulation -----
elif selected_menu == "Simulation":
    st.subheader("Simulation d’un nouveau scénario")

    # Récupération des IDs clients depuis l’API
    client_ids = get_all_client_ids()
    # Sélection d’un seul ID client (élimination du doublon)
    client_id = st.selectbox("Sélectionner un ID client", sorted(client_ids))
    # Récupération des données du client
    client_data = get_client_data(client_id)
    # Conversion en DataFrame
    original_data = pd.DataFrame([client_data])
    row = original_data.iloc[0]

    st.markdown("### Modifiez les caractéristiques du client pour simuler un scénario")
    st.markdown("## 🧍 Informations personnelles")
    CODE_GENDER = st.radio("Sexe", [0, 1], index=int(original_data["CODE_GENDER"].values[0]), format_func=lambda x: "Homme" if x == 0 else "Femme")
    FLAG_OWN_CAR = st.radio("Possède une voiture ?", [0, 1], index=int(original_data["FLAG_OWN_CAR"].values[0]), format_func=lambda x: "Non" if x == 0 else "Oui")
    FLAG_OWN_REALTY = st.radio("Possède un bien immobilier ?", [0, 1], index=int(original_data["FLAG_OWN_REALTY"].values[0]), format_func=lambda x: "Non" if x == 0 else "Oui")
    CNT_CHILDREN = st.number_input("Nombre d'enfants", min_value=0, max_value=3, step=1, value=int(original_data["CNT_CHILDREN"].values[0]))
    CNT_FAM_MEMBERS = st.number_input("Nombre de membres dans la famille", min_value=1, max_value=20, step=1, value=int(original_data["CNT_FAM_MEMBERS"].values[0]))
    AGE = st.number_input("Âge", min_value=18, max_value=100, value=int(original_data["AGE"].values[0]))
    # Statut familial
    family_status = display_one_hot_selectbox(row, "NAME_FAMILY_STATUS_", "Statut familial")
    # Type de logement
    housing = display_one_hot_selectbox(row, "NAME_HOUSING_TYPE_", "Type de logement")

    st.markdown("## 💰 Revenus & Crédit")   
    AMT_INCOME_TOTAL = st.number_input("Revenu total", min_value=0.0, value=float(original_data["AMT_INCOME_TOTAL"].values[0]))
    AMT_CREDIT = st.number_input("Montant du crédit", min_value=0.0, value=float(original_data["AMT_CREDIT"].values[0]))
    AMT_ANNUITY = st.number_input("Montant de l'annuité", min_value=0.0, value=float(original_data["AMT_ANNUITY"].values[0]))
    AMT_GOODS_PRICE = st.number_input("Prix des biens", min_value=0.0, value=float(original_data["AMT_GOODS_PRICE"].values[0]))
    INCOME_PER_PERSON = st.number_input("Revenu par personne", min_value=0.0, value=float(original_data["INCOME_PER_PERSON"].values[0]))
    ANNUITY_INCOME_PERC = st.number_input("Annuité / Revenu", min_value=0.0, value=float(original_data["ANNUITY_INCOME_PERC"].values[0]))
    PAYMENT_RATE = st.number_input("Taux de paiement", min_value=0.0, value=float(original_data["PAYMENT_RATE"].values[0]))
    CREDIT_TERM = st.number_input("Durée du crédit", min_value=0.0, value=float(original_data["CREDIT_TERM"].values[0]))
    contract_type= display_one_hot_selectbox(row,"NAME_CONTRACT_TYPE_","Type de contrat")

    st.markdown("## 🏙️ Région")
    # <span> est un élément HTML pour styliser du texte.
    # color: couleur du texte, font-size: taille
    # strong = gras
    # <br> balise de saut de ligne HTML
    st.markdown(
        """ 
        <span style='color:gray; font-size: 0.9em;'> 
        ℹ️ <strong>REGION_POPULATION_RELATIVE</strong> : proportion de la population vivant dans la région du client par rapport à la population totale. <br>
        Valeurs proches de 0 = région peu peuplée, proches de 1 = très peuplée.
        </span>
        """, unsafe_allow_html=True)
    REGION_POPULATION_RELATIVE = st.number_input("Population relative de la région", min_value=0.0, value=float(original_data["REGION_POPULATION_RELATIVE"].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>REGION_RATING_CLIENT_W_CITY</strong> : évaluation (1 à 3) de la qualité de la région où vit le client (1 = mauvaise, 3 = bonne).
        </span>
        """, unsafe_allow_html=True)
    REGION_RATING_CLIENT_W_CITY = st.selectbox("Note de la région", [1, 2, 3], index=[1, 2, 3].index(int(original_data["REGION_RATING_CLIENT_W_CITY"].values[0])))

    st.markdown("## 🧮 Scores externes")
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>EXT_SOURCE_1 / 2 / 3</strong> : scores de risque externes (fournis par des sources tierces). <br>
        Plus le score est élevé (proche de 1), plus le client est considéré comme fiable.
        </span>
        """, unsafe_allow_html=True)
    EXT_SOURCE_1 = st.number_input("Source externe 1", min_value=0.0, max_value=1.0, value=float(original_data["EXT_SOURCE_1"].values[0]))
    EXT_SOURCE_2 = st.number_input("Source externe 2", min_value=0.0, max_value=1.0, value=float(original_data["EXT_SOURCE_2"].values[0]))
    EXT_SOURCE_3 = st.number_input("Source externe 3", min_value=0.0, max_value=1.0, value=float(original_data["EXT_SOURCE_3"].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>YEARS_BUILD_AVG</strong> : année moyenne de construction des bâtiments de la zone, normalisée. <br>
        Valeur proche de 0 = anciens bâtiments, proche de 1 = récents.
        </span>
        """, unsafe_allow_html=True)
    YEARS_BUILD_AVG = st.number_input("Année moyenne de construction", min_value=0.0, value=float(original_data["YEARS_BUILD_AVG"].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>TOTALAREA_MODE</strong> représente la surface totale du logement du client, normalisée entre 0 et 1. <br>
        Par exemple : 0.1 ≈ petit logement (~20 m²), 0.5 ≈ moyen (~100 m²), 0.9 ≈ grand logement (~180 m²).
        </span>
        """, 
        unsafe_allow_html=True)
    TOTALAREA_MODE = st.number_input("Surface totale", min_value=0.0, value=float(original_data["TOTALAREA_MODE"].values[0]))

    st.markdown("## 🏦 Historique bancaire et crédit")
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>BURO_DAYS_CREDIT_MEAN</strong> : ancienneté moyenne des crédits passés (en jours). <br>
        Valeurs négatives (ex. -1500) indiquent l’ancienneté par rapport à aujourd’hui.
        </span>
        """, unsafe_allow_html=True)
    BURO_DAYS_CREDIT_MEAN = st.number_input("BURO: Jours de crédit moyen", value=float(original_data['BURO_DAYS_CREDIT_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>BURO_AMT_CREDIT_SUM_MEAN</strong> : montant moyen des crédits passés enregistrés dans les bureaux de crédit.
        </span>
        """, unsafe_allow_html=True)
    BURO_AMT_CREDIT_SUM_MEAN = st.number_input("BURO: Montant crédit moyen", value=float(original_data['BURO_AMT_CREDIT_SUM_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>BURO_AMT_CREDIT_SUM_OVERDUE_MEAN</strong> : montant moyen des crédits en retard (non remboursés à temps).
        </span>
        """, unsafe_allow_html=True)    
    BURO_AMT_CREDIT_SUM_OVERDUE_MEAN = st.number_input("BURO: Crédit en retard", value=float(original_data['BURO_AMT_CREDIT_SUM_OVERDUE_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>BURO_CREDIT_DAY_OVERDUE_MEAN</strong> : durée moyenne des retards de paiement (en jours).
        </span>
        """, unsafe_allow_html=True)
    BURO_CREDIT_DAY_OVERDUE_MEAN = st.number_input("BURO: Jours de retard moyen", value=float(original_data['BURO_CREDIT_DAY_OVERDUE_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>PREV_AMT_CREDIT_MAX</strong> : montant maximal accordé sur un crédit précédent.
        </span>
        """, unsafe_allow_html=True)
    PREV_AMT_CREDIT_MAX = st.number_input("Crédit précédent max", value=float(original_data['PREV_AMT_CREDIT_MAX'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>PREV_AMT_APPLICATION_MEAN</strong> : montant moyen demandé sur des crédits précédents.
        </span>
        """, unsafe_allow_html=True)
    PREV_AMT_APPLICATION_MEAN = st.number_input("Demande de crédit moyenne", value=float(original_data['PREV_AMT_APPLICATION_MEAN'].values[0])) 
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>PREV_APP_CREDIT_PERC_MEAN</strong> : ratio entre montant du crédit accordé et montant demandé. <br>
        Si &lt; 1, le client a obtenu moins que demandé.
        </span>
        """, unsafe_allow_html=True)
    PREV_APP_CREDIT_PERC_MEAN = st.number_input("Ratio montant crédit / demande", value=float(original_data['PREV_APP_CREDIT_PERC_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>PREV_DAYS_DECISION_MIN</strong> : date la plus ancienne (en jours) d’une décision de crédit passée.
        </span>
        """, unsafe_allow_html=True)
    PREV_DAYS_DECISION_MIN = st.number_input("Jours depuis décision précédente min", value=float(original_data['PREV_DAYS_DECISION_MIN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>POS_SK_DPD_MEAN</strong> : nombre moyen de jours de retard (DPD) sur les crédits en cours.
        </span>
        """, unsafe_allow_html=True)
    POS_SK_DPD_MEAN = st.number_input("POS: DPD moyen", value=float(original_data['POS_SK_DPD_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>POS_SK_DPD_DEF_MEAN</strong> : nombre moyen de jours de retard sérieux (défauts) sur POS.
        </span>
        """, unsafe_allow_html=True)
    POS_SK_DPD_DEF_MEAN = st.number_input("POS: Défaut moyen", value=float(original_data['POS_SK_DPD_DEF_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>INSTAL_PAYMENT_DIFF_MEAN</strong> : écart moyen entre paiement prévu et paiement réel.
        </span>
        """, unsafe_allow_html=True)
    INSTAL_PAYMENT_DIFF_MEAN = st.number_input("Diff. de paiement moyenne", value=float(original_data['INSTAL_PAYMENT_DIFF_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>INSTAL_PAYMENT_DIFF_MAX</strong> : écart maximum observé entre paiement prévu et réalisé.
        </span>
        """, unsafe_allow_html=True)
    INSTAL_PAYMENT_DIFF_MAX = st.number_input("Diff. de paiement max", value=float(original_data['INSTAL_PAYMENT_DIFF_MAX'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>INSTAL_DPD_MEAN</strong> : jours de retard moyen sur les échéances d’emprunt.
        </span>
        """, unsafe_allow_html=True)
    INSTAL_DPD_MEAN = st.number_input("INSTAL: DPD moyen", value=float(original_data['INSTAL_DPD_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>CC_AMT_BALANCE_MEAN</strong> : solde moyen des cartes de crédit.
        </span>
        """, unsafe_allow_html=True)
    CC_AMT_BALANCE_MEAN = st.number_input("CC: Solde moyen", value=float(original_data['CC_AMT_BALANCE_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN</strong> : montant moyen des paiements mensuels sur les cartes.
        </span>
        """, unsafe_allow_html=True)
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN = st.number_input("CC: Paiement total courant", value=float(original_data['CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>CC_SK_DPD_MAX</strong> : nombre maximum de jours de retard sur les cartes de crédit.
        </span>
        """, unsafe_allow_html=True)
    CC_SK_DPD_MAX = st.number_input("CC: DPD max", value=float(original_data['CC_SK_DPD_MAX'].values[0]))
    st.markdown(
        """
        <span style='color:gray; font-size: 0.9em;'>
        ℹ️ <strong>DAYS_EMPLOYED_PERC</strong> : ratio entre le nombre de jours travaillés et l'âge du client. <br>
        Permet d’estimer la stabilité professionnelle par rapport à l’âge (valeurs proches de 1 = long historique d’emploi).
        </span>
        """, unsafe_allow_html=True)
    DAYS_EMPLOYED_PERC = st.number_input("Jours employés en %", value=float(original_data['DAYS_EMPLOYED_PERC'].values[0]))

    st.markdown("## 💼 Informations professionnelles")
    # Type de revenu
    income_type=display_one_hot_selectbox(row,"NAME_INCOME_TYPE_","Type de revenu")
    # Education
    education_type=display_one_hot_selectbox(row,"NAME_EDUCATION_TYPE_","Niveau d'éducation")
    # Secteur
    sector = display_one_hot_selectbox(row, "SECTOR_", "Secteur d'activité")
    # Profession
    occupation = display_one_hot_selectbox(row,"OCCUPATION_","Profession")

# -----Variables catégorielles à encoder manuellement (one-hot encoding)-----
# -----Initialisation des colonnes avec 0-----
categorical_features = {  # Dictionnaire pour initialiser toutes les variables catégorielles à 0
    "NAME_CONTRACT_TYPE_Revolving loans": 0,
    "NAME_INCOME_TYPE_Commercial associate": 0,
    "NAME_INCOME_TYPE_Pensioner": 0,
    "NAME_INCOME_TYPE_State servant": 0,
    "NAME_INCOME_TYPE_Unemployed": 0,
    "NAME_INCOME_TYPE_Working": 0,
    "NAME_EDUCATION_TYPE_Higher education": 0,
    "NAME_EDUCATION_TYPE_Lower secondary": 0,
    "NAME_EDUCATION_TYPE_Secondary / secondary special": 0,
    "NAME_FAMILY_STATUS_Married": 0,
    "NAME_FAMILY_STATUS_Single / not married": 0,
    "NAME_FAMILY_STATUS_Widow": 0,
    "NAME_HOUSING_TYPE_House / apartment": 0,
    "NAME_HOUSING_TYPE_Office apartment": 0,
    "NAME_HOUSING_TYPE_Rented apartment": 0,
    "NAME_HOUSING_TYPE_With parents": 0,
    "SECTOR_Industry": 0,
    "SECTOR_Trade": 0,
    "SECTOR_Transport": 0,
    "SECTOR_Business Entity": 0,
    "SECTOR_Government": 0,
    "SECTOR_Security": 0,
    "SECTOR_Services": 0,
    "SECTOR_Construction": 0,
    "SECTOR_Medicine": 0,
    "SECTOR_Police": 0,
    "SECTOR_Other": 0,
    "OCCUPATION_Labor_Work": 0,
    "OCCUPATION_Sales_Services": 0,
    "OCCUPATION_Medical_Staff": 0,
    "OCCUPATION_Security": 0,
    "OCCUPATION_Management_Core": 0,
    "OCCUPATION_Other": 0
}

# -----Remplir les valeurs sélectionnées-----
categorical_features[f"NAME_CONTRACT_TYPE_{contract_type}"] = 1
categorical_features[f"NAME_INCOME_TYPE_{income_type}"] = 1
categorical_features[f"NAME_EDUCATION_TYPE_{education_type}"] = 1
categorical_features[f"NAME_FAMILY_STATUS_{family_status}"] = 1
categorical_features[f"NAME_HOUSING_TYPE_{housing}"] = 1
categorical_features[f"SECTOR_{sector}"] = 1
categorical_features[f"OCCUPATION_{occupation}"] = 1

# -----Rassembler les données modifiées dans un dictionnaire-----
# -----Fusionner les données modifiées numériques et catégorielles-----
client_data = pd.DataFrame([original_data.iloc[0]])  # conversion en DataFrame

data_for_api = client_data.to_dict(orient="records")[0] # Convertit le DataFrame en dictionnaire compatible API
data_for_api.update(categorical_features) #  Ajoute les variables catégorielles encodées
data_for_api.update({ # Ajoute les autres variables calculées ou extraites
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
    "CNT_CHILDREN": CNT_CHILDREN,
    "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
    "AGE": AGE,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    "INCOME_PER_PERSON": INCOME_PER_PERSON,
    "ANNUITY_INCOME_PERC": ANNUITY_INCOME_PERC,
    "PAYMENT_RATE": PAYMENT_RATE,
    "CREDIT_TERM": CREDIT_TERM,
    "REGION_POPULATION_RELATIVE": REGION_POPULATION_RELATIVE,
    "REGION_RATING_CLIENT_W_CITY": REGION_RATING_CLIENT_W_CITY,
    "EXT_SOURCE_1": EXT_SOURCE_1,
    "EXT_SOURCE_2": EXT_SOURCE_2,
    "EXT_SOURCE_3": EXT_SOURCE_3,
    "YEARS_BUILD_AVG": YEARS_BUILD_AVG,
    "TOTALAREA_MODE": TOTALAREA_MODE,
    "BURO_DAYS_CREDIT_MEAN": BURO_DAYS_CREDIT_MEAN,
    "BURO_AMT_CREDIT_SUM_MEAN": BURO_AMT_CREDIT_SUM_MEAN,
    "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN": BURO_AMT_CREDIT_SUM_OVERDUE_MEAN,
    "BURO_CREDIT_DAY_OVERDUE_MEAN": BURO_CREDIT_DAY_OVERDUE_MEAN,
    "PREV_AMT_CREDIT_MAX": PREV_AMT_CREDIT_MAX,
    "PREV_AMT_APPLICATION_MEAN": PREV_AMT_APPLICATION_MEAN,
    "PREV_APP_CREDIT_PERC_MEAN": PREV_APP_CREDIT_PERC_MEAN,
    "PREV_DAYS_DECISION_MIN": PREV_DAYS_DECISION_MIN,
    "POS_SK_DPD_MEAN": POS_SK_DPD_MEAN,
    "POS_SK_DPD_DEF_MEAN": POS_SK_DPD_DEF_MEAN,
    "INSTAL_PAYMENT_DIFF_MEAN": INSTAL_PAYMENT_DIFF_MEAN,
    "INSTAL_PAYMENT_DIFF_MAX": INSTAL_PAYMENT_DIFF_MAX,
    "INSTAL_DPD_MEAN": INSTAL_DPD_MEAN,
    "CC_AMT_BALANCE_MEAN": CC_AMT_BALANCE_MEAN,
    "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN": CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN,
    "CC_SK_DPD_MAX": CC_SK_DPD_MAX,
    "DAYS_EMPLOYED_PERC": DAYS_EMPLOYED_PERC
})

if st.button("🔍 Prédire la probabilité de défaut"):
     try:
        response = requests.post(f"{API_URL}/predict", json=data_for_api) # Appel API POST vers l'endpoint /predict
        response.raise_for_status()  # Déclenche une exception si le statut HTTP est une erreur
        result = response.json() # Récupère la réponse JSON
        proba = result["proba"] # Extrait la probabilité de défaut
        prediction = 1 if proba >= 0.10 else 0  # Seuil de 10% : au-dessus = refus, sinon acceptation

        st.markdown(f"### 🔢 Probabilité de défaut : **{proba:.2%}**")
        if prediction:
            st.error("❌ Crédit REFUSÉ (risque trop élevé)")
        else:
            st.success("✅ Crédit ACCEPTÉ (risque acceptable)")
     except requests.RequestException as e:
         st.error(f"Erreur lors de l'appel API : {e}")
