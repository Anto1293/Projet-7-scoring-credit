# Docker est une technologie qui permet de créer des "conteneurs" pour exécuter des applications de manière isolée. Cela garantit que l'application fonctionnera de la même manière, peu importe l'environnement où elle est déployée 

# API/Dockerfile

FROM python:3.10-slim

# Installer les bibliothèques système nécessaires
RUN apt-get update && apt-get install -y libgomp1

# Crée un dossier
WORKDIR /api

# Copie les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/best_model_lightgbm.pkl /model/best_model_lightgbm.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



