from sklearn.model_selection import train_test_split
import pandas as pd

# URL du dataset complet
DATA_URL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/train.parquet"

# Charger les données
df = pd.read_parquet(DATA_URL)

# Sous-échantillonnage stratifié (ex : 5000 lignes)
df_subset, _ = train_test_split(
    df,
    train_size=5000,
    stratify=df["TARGET"],  # Garde la proportion TARGET 0/1
    random_state=42
)

# Sauvegarde du fichier
df_subset.to_parquet("train_subset_with_target.parquet", index=False)


print("✅ Sous-échantillon de 5000 lignes créé et sauvegardé.")

