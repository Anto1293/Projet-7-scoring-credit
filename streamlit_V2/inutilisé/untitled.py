

palette = sns.color_palette("Accent")


def apply_labels(df, feature):
    """Applique les labels lisibles aux colonnes binaires si connues."""
    if feature in BINARY_LABELS:
        return df[feature].map(BINARY_LABELS[feature])  # Remplace 0/1 par "Oui"/"Non" si défini
    return df[feature]

st.subheader("📊 Comparaison Client")

# --- Sélection du client ---
client_ids = df_all_clients["SK_ID_CURR"].tolist()   # Liste des identifiants clients
selected_id = st.selectbox("Sélectionner un ID client", client_ids)   # Menu déroulant 
client_data = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id].squeeze()   # Extrait les données du client sélectionné (sous forme de série)

# --- Variable à comparer ---
all_features = df_all_clients.columns.drop("SK_ID_CURR")
feature = st.selectbox("📌 Choisir une variable à comparer", all_features, format_func=rename_variable) # Menu déroulant avec noms formatés

# --- Comparaison simple : tous les clients & groupe d'âge ---
col1, col2 = st.columns(2)
# Crée deux colonnes côte à côte
with col1:
    st.markdown("#### Répartition dans la population")
    fig, ax = plt.subplots(figsize=(5, 3))  # Création de la figure
    display_feature = apply_labels(df_all_clients, feature)  # Applique les labels
    sns.histplot(display_feature, bins=20, kde=False, ax=ax, color=palette[0])  # Histogramme
    client_val = apply_labels(client_data.to_frame().T, feature).values[0]  # Valeur client (transposé)
    ax.axvline(client_val, color=palette[4], linestyle="--", label="Client") # Ligne verticale pour le client
    ax.set_title(f"{rename_variable(feature)} - Tous les clients", fontsize=10)
    ax.set_xlabel(rename_variable(feature))
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("#### Comparaison dans votre groupe d’âge")
    df_all_clients["AGE_BIN"] = pd.cut(df_all_clients["AGE"], bins=[20, 30, 40, 50, 60, 70, 100], right=False)
    # Crée des tranches d'âge
    client_age_bin = df_all_clients[df_all_clients["SK_ID_CURR"] == selected_id]["AGE_BIN"].values[0]
    # Filtre les clients du même groupe d'âge
    group_age = df_all_clients[df_all_clients["AGE_BIN"] == client_age_bin]
    age_range = f"[{int(client_age_bin.left)} - {int(client_age_bin.right)})"  # Formate la tranche

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    display_feature_age = apply_labels(group_age, feature)
    sns.histplot(display_feature_age, bins=20, ax=ax2, color=palette[1])
    client_val_age = apply_labels(client_data.to_frame().T, feature).values[0]
    ax2.axvline(client_val_age, color=palette[4], linestyle="--", label="Client")
    ax2.set_title(f"{rename_variable(feature)} - Groupe d’âge {age_range}", fontsize=10)
    ax2.set_xlabel(rename_variable(feature))
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
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
    # Quantitatif vs Quantitatif
    sns.regplot(data=df_all_clients, x=x_feature, y=y_feature, ax=ax3, scatter=False, color=palette[2])
    # Régression linéaire (relation entre x et y et zone grise = marge d'erreur)
    ax3.scatter(client_data[x_feature], client_data[y_feature], color=palette[4], label="Client", s=40)
    ax3.set_title(f"{rename_variable(y_feature)} en fonction de {rename_variable(x_feature)}")
    ax3.set_xlabel(rename_variable(x_feature))
    ax3.set_ylabel(rename_variable(y_feature))

elif not is_x_num and is_y_num:
    # Catégoriel vs Quantitatif
    x_vals = apply_labels(df_all_clients, x_feature)
    sns.barplot(x=x_vals, y=df_all_clients[y_feature], estimator=np.mean, ax=ax3, palette=palette)
    ax3.scatter(
        [apply_labels(client_data.to_frame().T, x_feature).values[0]],
        [client_data[y_feature]],
        color=palette[4], label="Client")
    ax3.set_title(f"Moyenne de {rename_variable(y_feature)} par {rename_variable(x_feature)}")
    ax3.set_xlabel(rename_variable(x_feature))
    wrapped_label = textwrap.fill(f"Moyenne de {rename_variable(y_feature)}", width=25)
    ax3.set_ylabel(wrapped_label)


elif is_x_num and not is_y_num:
    # Quantitatif vs Catégoriel
    y_vals = apply_labels(df_all_clients, y_feature)
    sns.barplot(y=y_vals, x=df_all_clients[x_feature], estimator=np.mean, ax=ax3, orient="h", palette=palette)
    ax3.scatter(
        [client_data[x_feature]],
        [apply_labels(client_data.to_frame().T, y_feature).values[0]], color=palette[4], label="Client")
    ax3.set_title(f"Moyenne de {rename_variable(x_feature)} par {rename_variable(y_feature)}")
    ax3.set_ylabel(rename_variable(y_feature))
    ax3.set_xlabel(rename_variable(x_feature))
    
elif not is_x_num and not is_y_num:
    # Catégoriel vs Catégoriel
    x_vals = apply_labels(df_all_clients, x_feature)
    y_vals = apply_labels(df_all_clients, y_feature)
    ct = pd.crosstab(x_vals, y_vals, normalize='index') * 100   # Crosstab en %
    bars = ct.plot(kind="bar", stacked=True, ax=ax3, colormap="Accent")
    for container in bars.containers:
        bars.bar_label(container, fmt="%.0f%%", label_type="center", fontsize=8)   # % sur les barres au milieu
    ax3.set_title(f"Répartition (%) de {rename_variable(y_feature)} selon {rename_variable(x_feature)}", fontsize=10)
    ax3.set_xlabel(rename_variable(x_feature))
    ax3.set_ylabel("Pourcentage (%)")
    ax3.legend(title=rename_variable(y_feature), fontsize=6, title_fontsize=7)

else:
    st.warning("Ce type de croisement n’est pas encore géré.")
    
# Ajustement final
ax3.tick_params(axis='x', labelrotation=45)
ax3.tick_params(axis='y', labelrotation=45)
fig3.tight_layout()
st.pyplot(fig3)






