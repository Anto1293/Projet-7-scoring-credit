import streamlit as st

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
