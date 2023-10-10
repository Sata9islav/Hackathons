import pandas as pd
import datetime as dt


def datamart(ST: list[str], SKU: list[str],
             date=dt.datetime.now(),
             data_path='db\sales_df_train.csv',
             days_ago=14) -> tuple:
    """
   Функция принимает список магазинов и список товаров, для которых требуется предсказание

   Args:
      - ST - id магазина
      - SKU - id товара
      - data - дата, от которой необходимо строить предсказание на 2 недели
      - data_path - путь к csv-базе
      - days_ago - число предыдущих дней от заданой даты для сбора датасета

   Return:
      - датафрейм
   """

    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d')

    # отнимаем время от даты
    date = date - dt.timedelta(days=days_ago)

    # Вариант без базы данных
    dataframe = pd.read_csv(data_path)
    dataframe = dataframe[['st_id', 'pr_sku_id', 'date', 'pr_sales_in_units', 'pr_promo_sales_in_units']]

    # значения для мерджа
    store_item_request = {store: SKU for store in ST}
    table_for_merge = pd.DataFrame(store_item_request).T.reset_index()
    table_for_merge = table_for_merge.melt(id_vars='index') \
        .drop('variable', axis=1) \
        .rename(columns={'index': 'st_id', 'value': 'pr_sku_id'})

    # мердж нужных магазинов с БД
    dataframe = dataframe.merge(table_for_merge, how='right', on=['st_id', 'pr_sku_id'])

    # проверка на пустоту датасета для неизвестных продуктов или магазинов

    if dataframe['pr_sales_in_units'].dropna().shape[0] == 0:
        return 'none_info', dataframe

    dataframe['date'] = pd.to_datetime(dataframe['date'], format='%Y-%m-%d')
    dataframe = dataframe[dataframe['date'] >= date]
    dataframe['pr_without_promo_sales_in_units'] = dataframe['pr_sales_in_units'] - dataframe['pr_promo_sales_in_units']
    dataframe = dataframe.drop(['pr_sales_in_units', 'pr_promo_sales_in_units'], axis=1)
    dataframe = dataframe.sort_values(['date', 'st_id', 'pr_sku_id'])

    # проверка на отсутствие магазинов или товаров

    if dataframe[~dataframe['pr_without_promo_sales_in_units'].isna() == True].shape[0] != dataframe.shape[0]:
        return 'not_full_info', dataframe
    else:
        return 'full_info', dataframe
