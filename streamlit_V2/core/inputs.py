# core/inputs.py

import streamlit as st
from core.features import NUMERICAL_FEATURES, BINARY_FEATURES, OHE_MAPPING, FEATURE_GROUPS, FEATURE_DESCRIPTIONS
from core.predictions_utils import display_one_hot_selectbox
from core.features import rename_variable

def build_data_for_api(numerical_inputs, ohe_inputs, original_row):
    """
    Combine les inputs numériques et OHE dans une ligne complète compatible avec l'API.
    """
    data = original_row.copy()
    data.update(numerical_inputs)
    data.update(ohe_inputs)
    return data.to_dict()


def display_grouped_inputs(data_row):
    """
    Affiche tous les champs dans des sections regroupées selon FEATURE_GROUPS.
    Détecte automatiquement le type du champ (numérique, binaire, OHE).
    Affiche une info-bulle si présente.
    """
    all_values = {}

    for group_title, features in FEATURE_GROUPS.items():
        with st.expander(group_title):
            for feature in features:
                readable_label = rename_variable(feature)

                # Info-bulle si présente
                if feature in FEATURE_DESCRIPTIONS:
                    st.markdown(f"""
                    <span style='color:gray; font-size: 0.9em;'>
                    ℹ️ <strong>{feature}</strong> : {FEATURE_DESCRIPTIONS[feature]}
                    </span>
                    """, unsafe_allow_html=True)

                # 1. Champs binaires
                if feature in BINARY_FEATURES:
                    if feature == "CODE_GENDER":
                        val = "Femme" if data_row[feature] == 1 else "Homme"
                        selected = st.radio(readable_label, ["Homme", "Femme"], index=0 if val == "Homme" else 1)
                        all_values[feature] = 1 if selected == "Femme" else 0
                    else:
                        label = {
                            "FLAG_OWN_CAR": "Possède une voiture",
                            "FLAG_OWN_REALTY": "Possède un bien immobilier",
                            "NAME_CONTRACT_TYPE_Revolving_loans": "Revolving loans"
                        }.get(feature, readable_label)
                        val = bool(data_row[feature])
                        selected = st.radio(label, ["Non", "Oui"], index=1 if val else 0)
                        all_values[feature] = 1 if selected == "Oui" else 0

                # 2. Champs One-Hot Encoded
                elif feature in OHE_MAPPING:
                    prefix = OHE_MAPPING[feature]
                    selected = display_one_hot_selectbox(data_row, prefix, label=readable_label)
                    # Reset all to 0
                    for col in data_row.index:
                        if col.startswith(prefix):
                            all_values[col] = 0
                    # Set selected to 1
                    target_col = prefix + selected
                    if target_col in data_row.index:
                        all_values[target_col] = 1

                # 3. Champs à valeurs discrètes (ex: REGION_RATING)
                elif feature == "REGION_RATING_CLIENT_W_CITY":
                    val = int(data_row[feature])
                    selected = st.selectbox(readable_label, [1, 2, 3], index=[1, 2, 3].index(val))
                    all_values[feature] = selected

                # 4. Champs numériques
                else:
                    val = float(data_row.get(feature, 0.0))
                    new_val = st.number_input(readable_label, value=val)
                    all_values[feature] = new_val

    return all_values
