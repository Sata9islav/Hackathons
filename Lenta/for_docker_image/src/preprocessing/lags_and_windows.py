import pandas as pd
import numpy as np
import datetime as dt


def get_features_dataframe(dataframe: pd.core.frame.DataFrame, target_column: str, lags: list=[1],
                           moving_average: list = [1]) -> pd.core.frame.DataFrame:
    """
    Функция принимает датафрейм и создает новые признаки с помощью скользящего окна

    Args: 
        - dataframe - pandas dataframe
        - target_column - колонка, по которой будут строится признаки
        - lags - список лагов, т.е. значений признака за n-ный день
        - moving_average - список периодов (дней) для скользящего окна

    Return:
        - pandas dataframe с добавленными признаками
    """

    for lag in lags:
        dataframe[f'lag_{lag}_day'] = dataframe[target_column].shift(lag, fill_value=-1)
    if len(moving_average) > 0:
        for number in moving_average:
            dataframe[f'ewmavg_{number}_day'] = dataframe[target_column].ewm(span=number).mean().fillna(-1)
            dataframe[f'mavg_{number}_day'] = dataframe[target_column].rolling(number, center=False).mean().fillna(-1)
            dataframe[f'mmin_{number}_day'] = dataframe[target_column].rolling(number, center=False).min().fillna(-1)
            dataframe[f'mmax_{number}_day'] = dataframe[target_column].rolling(number, center=False).max().fillna(-1)
            dataframe[f'diff_{number}_day'] = dataframe[target_column].diff(number).fillna(-1)
    return dataframe


def add_target_features(stock_dataframe: pd.core.frame.DataFrame,  target_col: str, id_col: str, date_col: str,
                        lags: list[1], moving_average: list[1], target_log: bool = False) -> pd.core.frame.DataFrame:
    """
    Функция создает новые признаки для множественных временных рядов на основе уникального ID

    Args:
        - stock_dataframe - pandas dataframe
        - target_col - колонка, по которой будут строится признаки
        - id_col - колонка с уникальными ID
        - lags - список лагов, т.е. значений признака за n-ный день
        - moving_average - список периодов (дней) для скользящего окна
        - target_log - булево значения для логарифмирования признака с добавленной единицей
    
    Return:
        - pandas dataframe с добавленными признаками
    """

    new_dataframe = []
    for uniq_id in stock_dataframe[id_col].unique():
        temp_data = stock_dataframe[stock_dataframe[id_col] == uniq_id]
        temp_data = stock_dataframe.sort_values([date_col, id_col])
        if target_log:
            temp_data[target_col] = np.log1p(temp_data[target_col])
        temp_data = get_features_dataframe(temp_data,
                                           target_col,
                                           lags=lags,
                                           moving_average=moving_average)
        new_dataframe.append(temp_data)
    new_dataframe = pd.concat(new_dataframe)
    return new_dataframe
