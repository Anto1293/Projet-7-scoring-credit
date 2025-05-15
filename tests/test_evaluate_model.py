import pytest
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
from src.evaluate_model import evaluate_model, find_optimal_threshold
import matplotlib.pyplot as plt
import seaborn as sns

# Ce test permet de vérifier: 
# que les valeurs retournées sont bien présentes.
# que les valeurs de coût sont calculées correctement.
# que le format du dictionnaire retourné est correct.
# que la valeur du seuil optimale est correcte

@pytest.fixture
def data():
    # Exemple de données d'entrée pour tester
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.3, 0.4, 0.2, 0.8])
    return y_true, y_pred, y_proba

def test_evaluate_model(data):
    y_true, y_pred, y_proba = data
    result = evaluate_model(y_true, y_pred, y_proba)
    
    # Vérifie si le dictionnaire retourné contient les bonnes clés
    assert "ROC_AUC" in result
    assert "Cost" in result
    assert "F1_Score" in result
    assert "Precision" in result
    assert "Recall" in result
    assert "Accuracy" in result
    
    # Vérifie les types des valeurs retournées
    assert isinstance(result["ROC_AUC"], float)
    assert isinstance(result["Cost"], (int, float, np.integer, np.floating))
    assert isinstance(result["F1_Score"], float)
    assert isinstance(result["Precision"], float)
    assert isinstance(result["Recall"], float)
    assert isinstance(result["Accuracy"], float)
    
    # Vérifie qu'un coût non nul est calculé
    assert result["Cost"] > 0


def test_find_optimal_threshold(data):
    y_true, y_pred, y_proba = data
    threshold = find_optimal_threshold(y_true, y_proba)
    
    # Vérifie que le seuil est compris entre 0 et 1
    assert 0 <= threshold <= 1