import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_lgbm_importance(model, X, top_n=20):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(top_n), x='Importance', y='Feature', color='blue')
    plt.title("Top 20 des variables importantes (LightGBM)")
    plt.tight_layout()
    plt.show()

def shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X, plot_type="bar")

def shap_local(model, X, index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[1][index] if isinstance(shap_values, list) else shap_values[index],
        base_values=explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
        data=X.iloc[index],
        feature_names=X.columns
    ))
