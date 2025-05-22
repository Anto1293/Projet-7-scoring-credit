# ----- Fonctions API -----

import requests

#API endpoint
API_URL = "https://projet-7-scoring-credit.onrender.com"

def get_all_client_ids():
    return requests.get(f"{API_URL}/client/").json()

def get_client_data(client_id):
    return requests.get(f"{API_URL}/client/{client_id}").json()

def get_prediction(client_data):
    return requests.post(f"{API_URL}/predict", json=client_data).json()