# Fiche de description de l'architecture de l'API

## 1. Informations générales
Nom : API Scoring Client

Version : 1.0

Framework : FastAPI (Python)

Type d'API : RESTful

Protocole : HTTP/HTTPS

Format des données : JSON (entrée et sortie)

## 2. Endpoints
✅ HEAD /
Description : Vérifie la disponibilité de l'API (healthcheck).

Réponse : 200 OK (sans corps)

✅ GET /
Description : Message d'accueil.

Réponse :
{
  "message": "Bienvenue sur l'API de scoring client. Utilisez /predict pour prédire."
}
✅ GET /client/
Description : Retourne la liste des identifiants client (SK_ID_CURR).

✅ GET /client/{client_id}
Description : Retourne les données d’un client spécifique.

Paramètre : client_id (int)

Réponses :

200 OK : dictionnaire de features du client

404 Not Found : si le client est introuvable

✅ POST /predict
Description : Prédiction de la probabilité de défaut de paiement.

Corps de la requête : JSON selon le schéma InputData

Réponse :
{
  "proba": 0.0654,
  "décision": "accepté"
}
Seuil de décision : 0.10

## 3. Modèle de machine learning
Type : LightGBM

Chargement :

Par défaut via MLflow Model Registry (paramétrable via .env)

En fallback, chargement d’un modèle .pkl local via joblib

URI de tracking MLflow : défini par la variable MLFLOW_TRACKING_URI

## 4. Données utilisées
Source : dataset public Hugging Face au format .parquet 

Chargement d’un sous-échantillon sans la colonne TARGET au démarrage 

Colonne clé primaire : SK_ID_CURR

## 5. Gestion des erreurs
404 Not Found pour client non trouvé

400 Bad Request en cas d’erreur de prédiction ou format invalide

Utilisation de HTTPException standard de FastAPI
