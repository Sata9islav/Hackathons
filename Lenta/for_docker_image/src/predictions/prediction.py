import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model

MODEL_PATH = 'predictions/tr_models'
DF_OPTION_PATH = 'predictions/result_train_model.csv'
df_result = pd.read_csv(DF_OPTION_PATH)


def get_predictions(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    submission_with_predict = pd.DataFrame()
    unique_st_sku = df['st_sku'].unique()
    for i in unique_st_sku:
        print(i)
        l = load_model('predictions/tr_models/' + str(i), verbose=False)
        features = df_result[df_result['st_sku'] == i]['features'].to_list()

        df_subset = df[df['st_sku'] == i][features[0].split(',')]
        df_subset.drop(['pr_without_promo_sales_in_units'], axis=1, inplace=True)
        p = predict_model(l, data=df_subset, verbose=False)
        p['prediction_label'] = p['prediction_label'].apply(lambda x: 0 if x < 1 else x)
        p['prediction_label'] = p['prediction_label'].apply(lambda x: np.round(x))
        p['st_id'] = df.query('st_sku == @i')['st_sku'].apply(lambda x: str(x)[0:33])
        p['pr_sku_id'] = df.query('st_sku == @i')['st_sku'].apply(lambda x: str(x)[33:])
        p['date'] = df.query('st_sku == @i')['date'].values
        submission_with_predict = pd.concat([submission_with_predict, p])
    return submission_with_predict
