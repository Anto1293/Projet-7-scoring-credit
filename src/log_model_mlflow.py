import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def log_model_to_mlflow(run_name, model_name, model, params, metrics, y_true, y_pred, save_model=True):
    """
    Log d'un modèle entraîné dans MLflow avec ses paramètres, métriques, artefacts et modèle.
    """
    with mlflow.start_run(run_name=run_name):
        # Paramètres
        mlflow.log_param("model", model_name)
        mlflow.log_param("smote", True)
        mlflow.log_param("seuil", params.get("threshold", 0.5))
        if 'threshold' in params:
            del params['threshold']
        mlflow.log_params(params)

        # Métriques
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Modèle
        if save_model:
            mlflow.sklearn.log_model(model, "model")

        # Artefact : matrice de confusion
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
        plt.title(f"Matrice de confusion - {model_name}")
        fig_path = f"{run_name}_conf_matrix.png"
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        plt.close(fig)
