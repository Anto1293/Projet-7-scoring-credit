import pandas as pd
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataSummaryPreset
from evidently.presets import DataDriftPreset
import os

# Charger les jeux de données
df_train = pd.read_csv("../bdd/app_train_clean.csv")
df_test_final = pd.read_csv("../bdd/app_test_clean.csv")
df_train_final=df_train.drop(columns=['TARGET'])

# Résumé des datas
report_summary = Report([DataSummaryPreset()],include_tests="True")
my_eval = report_summary.run(df_train_final, df_test_final)
# Détection du datadrift
report_drift = Report([DataDriftPreset()],include_tests="True")
my_eval_drift= report_drift.run(df_train_final, df_test_final)

# Exporter HTML
my_eval_drift.save_html("my_eval_drift.html")
my_eval.save_html("my_eval.html")
print("Rapports sommaire et de datadrift sauvegardés")
