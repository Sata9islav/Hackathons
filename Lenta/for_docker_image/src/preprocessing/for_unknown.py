import pandas as pd
import numpy as np
import datetime as dt


def unknown_prediction(dataframe: pd.core.frame.DataFrame, date: str, days_predict: int) -> pd.core.frame.DataFrame:
    """
    Функция принимает dataframe, добавляет временной ряд и выставляет нулевые предсказания

    Args:
        - dataframe - датафрейм с неизвестными магазинами и/или товарами, для которых нет данных
        - data - дата, от которой строится предсказание
        - days_predict - сколько дней предсказать
    
    Return:
        - dataframe
    """
    # для нулевого предсказания требуются только эти две колонки
    dataframe = dataframe[['st_id', 'pr_sku_id']]
    # генерация временного ряда
    start_date = dt.datetime.strptime(date, '%Y-%m-%d')
    dates_array = np.arange(start_date, 
                            start_date + dt.timedelta(days=days_predict + 1),
                            dt.timedelta(days=1))
    dataframe.at[0, 'date'] = 0
    dataframe['date'] = dataframe['date'].astype('object')
    for row in range(dataframe.shape[0]):
        dataframe.at[row, 'date'] = dates_array
    dataframe = dataframe.explode('date')
    dataframe['prediction'] = 0
    return dataframe
