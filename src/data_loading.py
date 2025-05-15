from pathlib import Path
import pandas as pd

def load_all_data(data_dir):
    filenames = [
        'application_train.csv', 'application_test.csv', 'bureau_balance.csv', 'bureau.csv',
        'credit_card_balance.csv', 'installments_payments.csv', 'POS_CASH_balance.csv',
        'previous_application.csv', 'sample_submission.csv'
    ]
    dfs = {}
    for file in filenames:
        name = file.replace('.csv', '')
        dfs[name] = pd.read_csv(Path(data_dir) / file)
        print(f"{name} loaded: {dfs[name].shape}")
    return dfs
