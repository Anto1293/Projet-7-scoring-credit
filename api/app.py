from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import mlflow
import numpy as np
import os
import joblib
from dotenv import load_dotenv

load_dotenv()

# Charger le dataset une seule fois en mémoire (au démarrage de l'API)
DATA_URL = "https://huggingface.co/datasets/Antonine93/projet7scoring/resolve/main/train_subset_with_target.parquet"

# Charger un échantillon du dataset pour problème mémoire Render
df_full = pd.read_parquet(DATA_URL).drop(columns=["TARGET"])

# Vérifier si l'on utilise MLflow ou un modèle local
USE_MLFLOW = os.getenv("USE_MLFLOW", "true").lower() == "true"

if USE_MLFLOW:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    # Charger le modèle depuis MLflow (Model Registry)
    model_name = "LightGBM"
    model_alias = "final"
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
else:
    # Charger le modèle localement
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "..", "model", "best_model_lightgbm.pkl"))
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        raise Exception(f"Le fichier du modèle n'a pas été trouvé à {MODEL_PATH}")
        
print(f"MLFLOW_TRACKING_URI = {os.getenv('MLFLOW_TRACKING_URI')}")
print(f"USE_MLFLOW = {USE_MLFLOW}")

# Créer l'application FastAPI
app = FastAPI(title="API Scoring Client", description="Prédiction de défaut client")

# Endpoint HEAD
@app.head("/")
def head_root():
    # Retourne juste les headers, sans contenu
    return Response(status_code=200)

# Route d'accueil GET /
# GET / → retourne : {"message": "Bienvenue sur l'API de scoring client. Utilisez /score pour prédire."}
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de scoring client. Utilisez /predict pour prédire."}


@app.get("/client/")
def list_client_ids():
    """Endpoint pour accéder à la liste des ids crédits"""
    return df_full["SK_ID_CURR"].unique().tolist()


# Nouveau endpoint pour récupérer un client par ID
@app.get("/client/{client_id}")
def get_client_data(client_id: int):
    """Endpoint pour accéder aux données d'une demande de crédit précise"""
    filtered = df_full[df_full["SK_ID_CURR"] == client_id]
    if filtered.empty:
        raise HTTPException(status_code=404, detail="Client introuvable")
    return filtered.iloc[0].to_dict()


# Créer un schéma de données pour FastAPI (Input)
class InputData(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    CODE_GENDER: float
    FLAG_OWN_CAR: float
    FLAG_OWN_REALTY: float
    CNT_CHILDREN: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    REGION_POPULATION_RELATIVE: float
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT_W_CITY: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    YEARS_BUILD_AVG: float
    TOTALAREA_MODE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    IS_UNEMPLOYED: float
    BURO_DAYS_CREDIT_MEAN: float
    BURO_AMT_CREDIT_SUM_MEAN: float
    BURO_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    BURO_CREDIT_DAY_OVERDUE_MEAN: float
    PREV_AMT_CREDIT_MAX: float
    PREV_AMT_APPLICATION_MEAN: float
    PREV_APP_CREDIT_PERC_MEAN: float
    PREV_DAYS_DECISION_MIN: float
    POS_SK_DPD_MEAN: float
    POS_SK_DPD_DEF_MEAN: float
    INSTAL_PAYMENT_DIFF_MEAN: float
    INSTAL_PAYMENT_DIFF_MAX: float
    INSTAL_DPD_MEAN: float
    CC_AMT_BALANCE_MEAN: float
    CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN: float
    CC_SK_DPD_MAX: float
    DAYS_EMPLOYED_PERC: float
    INCOME_PER_PERSON: float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE: float
    CREDIT_TERM: float
    AGE: float
    NAME_CONTRACT_TYPE_Revolving_loans: float = Field(..., alias="NAME_CONTRACT_TYPE_Revolving loans")
    NAME_INCOME_TYPE_Commercial_associate: float = Field(..., alias="NAME_INCOME_TYPE_Commercial associate")
    NAME_INCOME_TYPE_Pensioner: float
    NAME_INCOME_TYPE_State_servant: float = Field(..., alias="NAME_INCOME_TYPE_State servant")
    NAME_INCOME_TYPE_Unemployed: float
    NAME_INCOME_TYPE_Working: float
    NAME_EDUCATION_TYPE_Higher_education: float = Field(..., alias="NAME_EDUCATION_TYPE_Higher education")
    NAME_EDUCATION_TYPE_Lower_secondary: float = Field(..., alias="NAME_EDUCATION_TYPE_Lower secondary")
    NAME_EDUCATION_TYPE_Secondary_special: float = Field(..., alias="NAME_EDUCATION_TYPE_Secondary / secondary special")
    NAME_FAMILY_STATUS_Married: float
    NAME_FAMILY_STATUS_Single_not_married: float = Field(..., alias="NAME_FAMILY_STATUS_Single / not married")
    NAME_FAMILY_STATUS_Widow: float
    NAME_HOUSING_TYPE_House_apartment: float = Field(..., alias="NAME_HOUSING_TYPE_House / apartment")
    NAME_HOUSING_TYPE_Office_apartment: float = Field(..., alias="NAME_HOUSING_TYPE_Office apartment")
    NAME_HOUSING_TYPE_Rented_apartment: float = Field(..., alias="NAME_HOUSING_TYPE_Rented apartment")
    NAME_HOUSING_TYPE_With_parents: float = Field(..., alias="NAME_HOUSING_TYPE_With parents")
    SECTOR_Industry: float
    SECTOR_Trade: float
    SECTOR_Transport: float
    SECTOR_Business_Entity: float = Field(..., alias="SECTOR_Business Entity")
    SECTOR_Government: float
    SECTOR_Security: float
    SECTOR_Services: float
    SECTOR_Construction: float
    SECTOR_Medicine: float
    SECTOR_Police: float
    SECTOR_Other: float
    OCCUPATION_Labor_Work: float
    OCCUPATION_Sales_Services: float
    OCCUPATION_Medical_Staff: float
    OCCUPATION_Security: float
    OCCUPATION_Management_Core: float
    OCCUPATION_Other: float

# Créer un endpoint POST /predict
@app.post("/predict")

def predict(data: InputData):
    try:
        # Convertir l'input en DataFrame
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])  # Correct avec Pydantic v2 au lieu de data.dict
        print("Colonnes envoyées au modèle :", input_df.columns.tolist())
        
        # Utiliser le modèle pour prédire
        prediction = model.predict(input_df)[0]
        
        # Appliquer le seuil optimal pour dire accepté / refusé
        decision = "refusé" if prediction > 0.10 else "accepté"
        
        return {
            "proba": round(float(prediction), 4),
            "décision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
