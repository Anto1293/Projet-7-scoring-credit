# 📊 Projet 7 - Modèle de Scoring Crédit

Ce projet a pour objectif de construire un modèle de scoring permettant de prédire la probabilité de défaut de paiement d’un client, et de déterminer si son crédit doit être accordé ou refusé.

## 🎯 Objectifs
- Construire un modèle de prédiction robuste
- Expliquer les prédictions (feature importance globale et locale)
- Mettre en production une API et son interface (via Streamlit)
- Suivre le datadrift avec Evidently
- Déployer des pytests automatiquement via GithubActions

## 🧰 Outils utilisés
- **Python**, Jupyter Notebook
- **LightGBM**, **Scikit-learn**, **SHAP**, **imbalanced-learn**
- **FastAPI**, **Streamlit**, **MLflow**, **Evidently**
- **Docker**, **Docker Compose**
- **Git**, **GitHub**, **GitHub Actions** pour le CI/CD

---
## 📂Données utilisées
Les fichiers de données proviennent du dataset Home Credit Default Risk :  
[https://www.kaggle.com/competitions/home-credit-default-risk/data]
---


## 🗂️ Structure du projet
.
├── docker-compose.yml

├── .env                          # Variables d'environnement (local)

├── requirements.txt

├── environment.yml

├── README.md

├── .github/
│   └── workflows/cicd.yml        # GitHub Actions : déploiement automatique

├── api/
│   ├── app.py                    # API FastAPI
│   └── Dockerfile

├── streamlit/
│   ├── streamlit_app.py          # Interface utilisateur
│   └── Dockerfile

├── streamlit_V2/
│   ├── pages                    # page 1, page 2, page 3 et page 4
│   ├── api                      # endpoints api
│   ├── core                     # fonctions
│   ├── interface_v2.py          # Interface utilisateur version améliorée (Projet 8)
│   └── Dockerfile

├── mlflow/
│   └── Dockerfile
│   └── artifacts/

├── model/
│   └── best_model_lightgbm.pkl

├── notebooks/
│   ├── Analyse exploratoire.ipynb
│   └── Model_scoring.ipynb

├── drift/
│   └── data_drift.py
│   └── my_eval_drift.html

├── src/
│   └── fonctions_utiles.py

└── tests/
    ├── test_evaluate_model.py
    ├── test_fast_api.py
    └── test_split_and_scale.py

## 🚀 Installation et exécution locale

### Cloner le repo
git clone [https://github.com/Anto1293/Projet-7-scoring-credit.git]

### Installer les dépendances
pip install -r requirements.txt

### Lancer l'API localement
cd api
uvicorn app:app --reload

### Tester avec Streamlit
cd streamlit
streamlit run streamlit_final.py

### Lancer les tests
pytest -s tests/

## Dépendances
Voir requirements.txt ou environment.yml

## Déploiement cloud
L’API est déployée sur [https://projet-7-scoring-credit.onrender.com]
Streamlit est accessible sur [https://projet-7-scoring-credit-interface.onrender.com]

## Docker Compose
docker-compose up --build
Cela lance : L'API (FastAPI), MLflow pour le suivi et l'application Streamlit

## Mise à jour
Interface streamlit version 2 accessible sur [https://projet-7-interface-version2.onrender.com]