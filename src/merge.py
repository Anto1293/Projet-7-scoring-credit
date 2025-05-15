import pandas as pd
def merge_aggregated_features(app_train, app_test, agg_dfs):
    """
    Merge des tables _agg en une seul table train et test
    """
    app_train = app_train.set_index('SK_ID_CURR')
    app_test = app_test.set_index('SK_ID_CURR')

    for agg_df in agg_dfs:
        app_train = app_train.join(agg_df, how='left')
        app_test = app_test.join(agg_df, how='left')

    app_train.reset_index(inplace=True)
    app_test.reset_index(inplace=True)

    return app_train, app_test
