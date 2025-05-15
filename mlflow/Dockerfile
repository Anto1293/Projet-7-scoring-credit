# mlflow/Dockerfile

FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer MLflow (et autres dépendances minimales)
RUN pip install --no-cache-dir mlflow

# Crée un dossier pour stocker les artefacts MLflow si jamais n'existe pas
RUN mkdir -p /mlflow/artifacts

# Copier les fichiers utiles
COPY . .

# Expose le port de l'interface MLflow
EXPOSE 5000

# Commande de démarrage
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
