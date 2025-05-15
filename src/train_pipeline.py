### train_models.py
from evaluate_model import evaluate_model, find_optimal_threshold

def train_pipeline(pipeline, X_train, y_train, X_val, y_val):
    pipeline.fit(X_train, y_train)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred_std = (y_val_proba >= 0.5).astype(int)
    evaluate_model(y_val, y_pred_std, y_val_proba, threshold=0.5)
    threshold_opt = find_optimal_threshold(y_val, y_val_proba)
    y_pred_opt = (y_val_proba >= threshold_opt).astype(int)
    evaluate_model(y_val, y_pred_opt, y_val_proba, threshold=threshold_opt)
    return pipeline, threshold_opt