import pandas as pd
import datetime as dt
from .data_mart import datamart
from .time_preprocess import add_time_features
from .lags_and_windows import add_target_features
from .for_unknown import unknown_prediction


def main(ST: list[str], SKU: list[str], DATE=dt.datetime.now(), PREDICT_DAYS=5) -> pd.core.frame.DataFrame:
    # проверка корректности введенной даты
    MAX_DAYS_AGO = 14
    LAGS = [1, 2, 7, 14]
    WINDOW_SIZE = [7, 14]
    DB_PATH = 'preprocessing/db/sales_df_train.csv'

    if isinstance(DATE, str):
        if dt.datetime.strptime(DATE, '%Y-%m-%d') > dt.datetime.now():
            return ('Введена некорректная дата. \n Дата не может быть позже сегодняшнего числа')
    else:
        if DATE > dt.datetime.now():
            return ('Введена некорректная дата. \n Дата не может быть позже сегодняшнего числа')
    # делаем срез из базы данных
    # st_id: list[str], sku_id: list[str],
    result, data = datamart(ST=ST, SKU=SKU, date=DATE, data_path=DB_PATH, days_ago=MAX_DAYS_AGO)[:10]
    # если все магазины и/или товары неизвестны, генерируем нулевые предсказания
    if result == 'none_info':
        data = unknown_prediction(dataframe=data, date=DATE, days_predict=PREDICT_DAYS)
        return data
    elif result == 'not_full_info':
        nan_data = data[data['pr_sales_in_units'].isna() == True]
        # эту таблица содержит нулевые предсказания
        nan_data = unknown_prediction(dataframe=nan_data, date=DATE, days_predict=PREDICT_DAYS)
        data = data.dropna()  # данные без Nan
    # добавляем временные признаки
    data = add_time_features(dataframe=data, st_id='st_id',
                             sku_id='pr_sku_id', col_name='st_sku',
                             date_col='date', drop=True,
                             value_types=['day', 'week', 'month', 'dow'])
    # добавляем признаки со скользящим окном
    data = add_target_features(stock_dataframe=data, target_col='pr_without_promo_sales_in_units',
                               id_col='st_sku', date_col='date',
                               lags=LAGS, moving_average=WINDOW_SIZE,
                               target_log=False)
    return data
