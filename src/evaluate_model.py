import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_proba, threshold=0.5):
    """
    Affiche les métriques principales d'évaluation du modèle et les retourne sous forme de dictionnaire.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = 10 * fn + 1 * fp
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n🔎 Seuil appliqué : {threshold:.2f}")
    print(f"🎯 Coût métier = {cost}")
    print(f"✅ F1-score = {f1:.4f}")
    print(f"✅ Précision = {precision:.4f}")
    print(f"✅ Rappel = {recall:.4f}")
    print(f"✅ ROC AUC = {roc_auc:.4f}")
    print(f"✅ Accuracy = {accuracy:.4f}")
    print(pd.Series(y_pred).value_counts())

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Matrice de confusion")
    plt.show()

    return {
        "ROC_AUC": roc_auc,
        "Cost": cost,
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }


def find_optimal_threshold(y_true, y_proba, cost_fn_weight=10, cost_fp_weight=1, step=0.05):
    """
    Trouve le seuil qui minimise le coût métier défini.
    """
    thresholds = np.arange(0.05, 1.0, step)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_fn_weight * fn + cost_fp_weight * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold


def plot_roc(y_true, y_proba, label=None):
    """
    Affiche la courbe ROC 
    (vrais positifs: mauvais clients dans mauvais/ faux positifs: bons clients dans mauvais)
    """
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, lw=2, label=f"{label or 'ROC'} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend()
    plt.grid(True)
    plt.show()
