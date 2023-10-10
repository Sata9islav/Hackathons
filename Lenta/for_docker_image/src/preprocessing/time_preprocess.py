import pandas as pd
import datetime as dt
import re
import numpy as np
from dateutil.easter import *
import pickle

with open('preprocessing/dictionaries/most_popluar_item.pkl', 'rb') as file:
    DAILY_DICT = pickle.load(file)

with open('preprocessing/dictionaries/item_dict.pkl', 'rb') as file:
    ITEM_DICT = pickle.load(file)

with open('preprocessing/dictionaries/store_dict.pkl', 'rb') as file:
    STORE_DICT = pickle.load(file)

with open('preprocessing/dictionaries/weight.pkl', 'rb') as file:
    WEIGHT_DICT = pickle.load(file)

with open('preprocessing/dictionaries/last_sale_type.pkl', 'rb') as file:
    LAST_SALE_TYPE = pickle.load(file)


def is_holiday(date: str) -> int:
    """
    Функция возвращает булево значение, если дата на входе
    совпадает с паттерном праздника
    """

    def is_easter():
        """
        Функция возвращает паттерн даты православной Пасхи для текущего года
        и нескольких дней до нее

        Пример, если Пасха 2024-05-05, то вернется '2024-05-[03|04|05]'
        """
        # определяем текущий год
        curr_year = dt.datetime.now().year
        easter_date = easter(curr_year, 2)
        # возьмем несколько дней до самой Пасхи
        # переведем их в строки и соединим логическим оператором ИЛИ
        easter_days = np.arange((easter_date - dt.timedelta(days=3)).day, (easter_date).day + 1)
        easter_days = [f'{i:02}' for i in easter_days]
        easter_days = '|'.join(easter_days)
        # выделим месяц и проведем те же операции
        easter_months = np.arange((easter_date - dt.timedelta(days=3)).month, (easter_date).month + 1)
        easter_months = [f'{i:02}' for i in easter_months]
        easter_months = '|'.join(easter_months)
        # объединяем в паттерн типа '2024-05-[03|04|05]'
        pattern = fr'{curr_year}-({easter_months})-({easter_days})'
        return pattern


    # паттерны для гос праздников
    december = r'20\d{2}-12-(2[4-9]|30|31)'
    january = r'20\d{2}-01-((0[1-9])|(1[0-3]))'
    february = r'20\d{2}-02-2[1-3]'
    march = r'20\d{2}-03-0[6-8]'
    may = r'20\d{2}-((04-(29|30))|(05-0[1-9]))'
    june = r'20\d{2}-06-1[0-2]'
    september = r'20\d{2}-((08-(28|30))|(09-01))'
    november = r'20\d{2}-11-0[2-4]'
    easter_date = is_easter()
    patterns = [december, january, february, march, may, june, september, november, easter_date]
    for pattern in patterns:
        # возвращаем полное совпадение
        string_date = date.strftime('%Y-%m-%d')
        res = re.match(pattern, string_date)
        if res:
            return 1
    return 0


def data_converter(dataframe: pd.core.frame.DataFrame,
                   column='date', value_types: list = ['day', 'week']) -> pd.core.frame.DataFrame:
    """
    Функция принимает на вход датафрейм, имя колонки с датой, список создаваемых признаков
    :: dataframe - pandas dataframe
    :: column - столбцец с датой формата "YYYY-mm-dd"
    :: value_types - значения (day, dow(day of week), dow(day of year), week, month, year)

    ::Return - pandas dataframe с новыми колонками
    """
    if dataframe[column].dtype == 'O':
        dataframe[column] = pd.to_datetime(dataframe[column], format='%Y-%m-%d')
    for value_type in value_types:
        if value_type == 'day':
            dataframe['day_sin'] = np.sin(2 * np.pi * dataframe[column].dt.day / dataframe[column].dt.days_in_month)
            dataframe['day_cos'] = np.cos(2 * np.pi * dataframe[column].dt.day / dataframe[column].dt.days_in_month)
        elif value_type == 'dow':
            dataframe['dow_sin'] = np.sin(2 * np.pi * dataframe[column].dt.dayofweek / 7)
            dataframe['dow_cos'] = np.cos(2 * np.pi * dataframe[column].dt.dayofweek / 7)
        elif value_type == 'doy':
            dataframe['doy_sin'] = np.sin(2 * np.pi * dataframe[column].dt.dayofyear / 365)
            dataframe['doy_cos'] = np.cos(2 * np.pi * dataframe[column].dt.dayofyear / 365)
        elif value_type == 'week':
            dataframe['week_sin'] = np.sin(2 * np.pi * dataframe[column].dt.isocalendar().week / 52)
            dataframe['week_cos'] = np.cos(2 * np.pi * dataframe[column].dt.isocalendar().week / 52)
        elif value_type == 'month':
            dataframe['month_sin'] = np.sin(2 * np.pi * dataframe[column].dt.month / 12)
            dataframe['month_cos'] = np.cos(2 * np.pi * dataframe[column].dt.month / 12)
        elif value_type == 'year':
            dataframe['year_number'] = dataframe[column].dt.year // 2020
    return dataframe


def get_daily_sells(sku_id: str) -> int:
    # если id товара есть в словаре, вернет 1, иначе 0
    result = DAILY_DICT.get(sku_id, 0)
    return result


def get_item_marker(sku_id: str) -> int:
    # если id товара есть в словаре, вернет 1, иначе 0
    result = ITEM_DICT.get(sku_id, -1)
    return result


def get_store_marker(st_id: str) -> int:
    # если id товара есть в словаре, вернет 1, иначе 0
    result = STORE_DICT.get(st_id, -1)

    return result


def convert_to_st_sku_id(dataframe: pd.core.frame.DataFrame,
                         st_id: str, sku_id: str,
                         col_name='st_sku', drop=True) -> pd.core.frame.DataFrame:
    # создание общего ключа: магазин-товар
    dataframe[col_name] = dataframe[st_id] + dataframe[sku_id]
    dataframe = dataframe.drop([st_id, sku_id], axis=1)

    return dataframe


def convert_weight(value):
    value = WEIGHT_DICT.get(value, -1)
    if value > 0:
        return 1 if value == 1 else 0
    return -1


def convert_store_type(store_type):
    return 1 / int(store_type)


def get_last_sale_type(value):
    value = LAST_SALE_TYPE.get(value, -1)
    if value > -1:
        return int(value)
    return value


def add_time_features(dataframe: pd.core.frame.DataFrame,
                      st_id='st_id', sku_id='pr_sku_id',
                      col_name='st_sku', date_col='date',
                      drop=True, value_types=['day', 'week', 'month', 'dow']) -> pd.core.frame.DataFrame:
    dataframe['is_holiday'] = dataframe['date'].apply(is_holiday)
    dataframe['daily_sells'] = dataframe['pr_sku_id'].apply(get_daily_sells)
    dataframe['st_markers'] = dataframe['st_id'].apply(get_store_marker)
    dataframe['sku_markers'] = dataframe['pr_sku_id'].apply(get_item_marker)
    dataframe['pr_uom_id'] = dataframe['pr_sku_id'].apply(convert_weight)
    dataframe = convert_to_st_sku_id(dataframe, st_id=st_id,
                                     sku_id=sku_id, col_name=col_name,
                                     drop=True)
    dataframe['pr_sales_type_id'] = dataframe['st_sku'].apply(get_last_sale_type)
    dataframe = data_converter(dataframe, column=date_col, value_types=value_types)
    return dataframe
