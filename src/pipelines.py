### pipelines.py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42

def get_logistic_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', LogisticRegression(solver='saga', class_weight='balanced', C= 0.50, penalty= "l2", random_state=RANDOM_STATE))])

def get_rf_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))
    ])

def get_lgbm_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', LGBMClassifier(learning_rate=0.11114350653481823,
                                      num_leaves=150,
                                      max_depth=12,
                                      min_child_samples=100,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      reg_alpha=1.0,
                                      reg_lambda=1.0,
                                      scale_pos_weight=1,
                                      class_weight='balanced',
                                      random_state=RANDOM_STATE,
                                      n_jobs=-1,
                                      verbose=-1))])

def gnb_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', GaussianNB())])


# Pipeline baseline (dummy)
 def baseline_pipeline():
     return = ImbPipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('smote', SMOTE(random_state=RANDOM_STATE)),
         ('classifier', DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE))])
