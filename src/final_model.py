### final_model.py
import pickle
import pandas as pd
from lightgbm import LGBMClassifier

def train_final_model(X_train, y_train, X_test, threshold=0.10):
    model = LGBMClassifier(
        colsample_bytree=0.5,
        learning_rate=0.1111,
        max_depth=12,
        min_child_samples=100,
        n_estimators=300,
        num_leaves=150,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.5,
        scale_pos_weight=1,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    with open("models/final_model_lightgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Modèle sauvegardé.")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    pd.Series(y_pred).value_counts().plot(kind="bar")
    return y_pred
