import argparse
from fastapi import FastAPI, Request
from pydantic import BaseModel
import preprocessing
from models import model
from typing import List
import uvicorn


class Item(BaseModel):
    sku: str
    count: int
    size1: str
    size2: str
    size3: str
    weight: str
    type: List[str]


class Order(BaseModel):
    orderId: str
    items: List[Item]


LIST_BIG_CARTON = ['STRETCH', 'NONPACK']
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pack")
def get_prediction(request: Order):
    items = []
    for el in request.items:
        items.append(el.dict())
    df_initial = preprocessing.read_data(items)
    df_order = preprocessing.sorted_dementions_sku(df_initial)
    df_for_clustering = preprocessing.get_df_for_cluster(df_order,
                                                         preprocessing.tfidf_vectorizer,
                                                         preprocessing.scaler_rb_for_cluster)
    df_order['cluster'] = model.kmeans.predict(df_for_clustering.to_numpy())

    dict_sku = {}
    # перебираем кластеры по очереди
    for cluster in sorted(df_order['cluster'].unique()):
        df_cluster = df_order[df_order['cluster'] == cluster]
        # Отсортируем df_cluster по объёму
        df_cluster['volume'] = df_cluster['a'] * df_cluster['b'] * df_cluster['c']
        df_cluster = df_cluster.sort_values('volume', ascending=False)
        df_cluster = df_cluster.drop('volume', axis=1)
        # сбросим индексы
        df_cluster = df_cluster.reset_index(drop=True)
        # Создадим словарь с текущей стопкой sku товаров
        list_sku = []
        while df_cluster.shape[0] != 0:
            index_for_drop = [0]  # для записи индексов товаров, которые упакованы
            # делаем текущим айтемом первую строку, добавляем её в текущий список sku
            item = df_cluster.iloc[:1]
            old_item = item.copy(deep=True)
            list_sku.append(item.loc[0]['sku'])
            # проводим предобработку sku в формат для предсказания и делаем прогноз
            item = preprocessing.get_df_for_predict(item, preprocessing.tfidf_vectorizer,
                                                    preprocessing.scaler_rb_for_clusific,
                                                    model.column_for_drop)
            predict = model.get_fit_predict_list_n(model.index, item)
            # Проверяем не один ли товар в ордере, если один, то предиктим, добавляем в словарь, и выходим из цикла
            if df_cluster.shape[0] == 1:
                dict_sku[' '.join(map(str, list_sku))] = predict
                break
            # Проверяем не был ли товар крупногабаритным по упаковке, если был, то просто добавляем в словарь и дропаем его из датасета
            if model.get_bool_list_ib_list(LIST_BIG_CARTON, predict):
                dict_sku[' '.join(map(str, list_sku))] = predict
                df_cluster = df_cluster[df_cluster.index != 0]
                df_cluster = df_cluster.reset_index(drop=True)
                list_sku = []
            else:
                # если товар не был крупногабаритным, то перебираем все оставшиеся товары, чтобы доукомплектовать коробку
                for i in range(1, df_cluster.shape[0]):
                    # Выбираем наименьший размер текущего айтема и доавляем кнему наименьший размер следующего
                    new_item = df_cluster.iloc[i:i + 1]
                    new_item_sku = new_item.loc[i]['sku']
                    new_item['a'] = old_item['a'].values + new_item['a'].values
                    # после этого переопределяем наибольший размер по оставшимся двум измерениям
                    new_item['b'] = max(old_item['b'].values, new_item['b'].values)
                    new_item['c'] = max(old_item['c'].values, new_item['c'].values)
                    # определяем вес объединенного айтема
                    new_item['goods_wght'] = old_item['goods_wght'].values + new_item['goods_wght'].values
                    # определяем новый карготип товара, для этого объединяем карготипы двух рассматриваемых товаров,
                    # разделяем их в список, находим уникальные вхождения при помощи множеств, объединяем обратно в строку
                    new_item['cargotype'] = ' '.join(
                        set((old_item.loc[0]['cargotype'] + new_item.loc[i]['cargotype']).split()))
                    # проводим предобработку sku в формат для предсказания и делаем прогноз
                    item_for_pred = preprocessing.get_df_for_predict(new_item.reset_index(drop=True),
                                                                     preprocessing.tfidf_vectorizer,
                                                                     preprocessing.scaler_rb_for_clusific,
                                                                     model.column_for_drop)
                    predict_list = model.get_fit_predict_list_n(model.index, item_for_pred)
                    # если новый товар не крупногабаритный, тогда добавляем его в список супер товара
                    # и дропаем второй товар из df_cluster
                    if not model.get_bool_list_ib_list(LIST_BIG_CARTON, predict_list):
                        list_sku.append(new_item_sku)
                        index_for_drop.append(i)
                        old_item = new_item.reset_index(drop=True)
                        # удаляем объект, обнуляем индексы
                df_cluster = df_cluster[~df_cluster.index.isin(index_for_drop)]
                df_cluster = df_cluster.reset_index(drop=True)
                old_item_for_pred = preprocessing.get_df_for_predict(old_item,
                                                                     preprocessing.tfidf_vectorizer,
                                                                     preprocessing.scaler_rb_for_clusific,
                                                                     model.column_for_drop)
                # После обхода добавляем в словарь очередной набор
                dict_sku[' '.join(map(str, list_sku))] = model.get_fit_predict_list_n(model.index, old_item_for_pred)
                list_sku = []
    return {"orderId": request.orderId,
            "package": dict_sku,
            "status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8001, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    # parser.add_argument("--debug", action="store_true", dest="debug")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
    uvicorn.run(app, host="0.0.0.0", port=8001, debug=True)
