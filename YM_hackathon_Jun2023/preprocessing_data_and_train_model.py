"""Этот скрипт обрабатывает исходные данные, обучает и выгружает модели, векторизаторы, масштабизаторы
   в папку для сборки докер образа."""

import logging
import numpy as np
import pandas as pd
import joblib
import os
import pickle
from progress.bar import FillingSquaresBar
import sys

import scipy
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from statistics import mean

import faiss
from faiss import write_index

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# основная директория
cwd = os.getcwd()
# путь к исходным данным
PATH_DATA = cwd + '/initial_data/data.csv'
PATH_SKU = cwd + '/initial_data/sku.csv'
PATH_SKU_CARGOTYPES = cwd + '/initial_data/sku_cargotypes.csv'
PATH_CARTON = cwd + '/initial_data/carton.csv'
PATH_CARGOTYPES_INFO = cwd + '/initial_data/cargotype_info.csv'
PATH_CARTON_PRICE = cwd + '/initial_data/carton_price.xlsx'
# пути к обработанным данным
PATH_TO_SAVE_DF_FOR_CLASSIFIC = cwd + '/preprocessed_data/df_for_classific.csv'
PATH_TO_SAVE_REC_PRED = cwd + '/preprocessed_data/prediction_rec_sys.csv'
PATH_TO_SAVE_SPARSE_FOR_CLUSTER = cwd + '/preprocessed_data/sparse_cluster.npz'
PATH_TO_SAVE_SPARSE_FOR_CLUSTER_TO_CLASSIF = cwd + '/preprocessed_data/sparse_cluster_to_classif.npz'
# пути для сохранения нормализаторов и векторайзера
PATH_TO_SAVE_SCALER_CLASIFIC = cwd + '/for_docker_images/src/scalers/scaler_rb_for_clusific.bin'
PATH_TO_SAVE_SCALER_FOR_CLUSTER = cwd + '/for_docker_images/src/scalers/scaler_rb_for_cluster.bin'
PATH_TO_SAVE_TFIDF_VECTORIZER = cwd + '/for_docker_images/src/vectorizers/tfidf_vectorizer.bin'
# пути для сохранения моделей и энкодера
PATH_TO_SAVE_ENCODER = cwd + '/for_docker_images/src/scalers/encoder.bin'
PATH_TO_SAVE_CLUSTERING_MODEL = cwd + "/for_docker_images/src/models/clustering_model.pkl"
PATH_TO_SAVE_FAISS = cwd + "/for_docker_images/src/models/fiass_index_with_drop_columns.index"
PATH_TO_SAVE_COLUMNS_TO_DROP = cwd + "/for_docker_images/src/preprocessed_data/columns_to_drop"
PATH_TO_SAVE_COLUMNS_TARGET = cwd + "/for_docker_images/src/preprocessed_data/target.csv"

# логирование
def start_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.CRITICAL)
    # FORMATTER = logging.Formatter('%(message)s')
    # console_handler.setFormatter(FORMATTER)
    # logger.addHandler(console_handler)
    path_log = os.path.join(cwd, 'Log_Page_Loader.log')
    file_handler = logging.FileHandler(path_log)
    file_handler.setLevel(logging.INFO)
    FORMATTER = logging.Formatter(
        '%(asctime)s  - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)


# Функции для предобработки данных
def cleaning_sku(df):
    df['list_demensions'] = df.apply(lambda x: sorted([x['a'], x['b'], x['c']]), axis=1)
    df['a'] = df.apply(lambda x: x['list_demensions'][0], axis=1)
    df['b'] = df.apply(lambda x: x['list_demensions'][1], axis=1)
    df['c'] = df.apply(lambda x: x['list_demensions'][2], axis=1)
    df = df[df['a'] < 175]
    df = df[df['b'] < 190]
    df = df[df['c'] < 500]
    return df


def get_one_carton(df_cart, name_column):
    # рассчёт уникальных вхождений вероятностей для каждого товара
    df_perc_car = df_cart.groupby(['sku', 'perc_carton'])['count_sku_cart'].agg('count').reset_index(drop=False)
    dict_count_perc = {sku: df_perc_car[df_perc_car['sku'] == sku]['sku'].count() for sku in
                       df_perc_car['sku'].unique()}
    df_cart['count_perc'] = df_cart['sku'].map(dict_count_perc)
    # Выбор для товаров у которых есть разные вхождения
    df = df_cart[df_cart['count_perc'] != 1]
    df = df.groupby('sku')['perc_carton'].agg('max').reset_index(drop=False)
    df_final = df.merge(df_cart[['sku', 'perc_carton', name_column, 'volume']], how='left', on=['sku', 'perc_carton'])
    # Обработка тех у кого дублируются максимальные вхождения
    dupl_sku = df_final[df_final['sku'].duplicated()]['sku']
    df_dupl_max = df_final[df_final['sku'].isin(dupl_sku)]
    df_final = df_final[~df_final['sku'].isin(dupl_sku)]
    # Выбор для товаров с одинаковой частотой вхождения
    df = df_cart[df_cart['count_perc'] == 1]
    # Добавим дублирующиеся максимальные вхождения
    df = pd.concat((df, df_dupl_max), axis=0)
    df = df.groupby('sku')['volume'].agg('min').reset_index(drop=False)
    df = df.merge(df_cart[['sku', 'perc_carton', 'volume', name_column]], how='left', on=['sku', 'volume'])
    df_final = pd.concat((df_final, df), axis=0)
    return df_final[['sku', name_column]]


def cleaning_data(df_data_in, df_carton_in, name_column):
    df_cart = df_carton_in.copy(deep=True)
    selected_carton_sku = df_data_in.groupby(['sku', name_column])['sku'].agg('count')
    df_sel_cart_sku = (pd.DataFrame(selected_carton_sku).
                       rename(columns={'sku': 'count_sku_cart'}).
                       reset_index(drop=False))
    df_count_sku = (pd.DataFrame(df_data_in.groupby('sku')['sku'].agg('count')).
                    rename(columns={'sku': 'count_sku_all'}).
                    reset_index(drop=False))
    df_sel_cart_sku = df_sel_cart_sku.merge(df_count_sku, how='left', on='sku')
    df_sel_cart_sku['perc_carton'] = df_sel_cart_sku['count_sku_cart'] / df_sel_cart_sku['count_sku_all']
    df_cart['volume'] = df_cart['LENGTH'] * df_cart['WIDTH'] * df_cart['HEIGHT']
    df_cart = df_cart.rename(columns={'CARTONTYPE': name_column})
    df_sel_cart_sku = df_sel_cart_sku.merge(df_cart[[name_column, 'volume']], how='left', on=name_column)
    return get_one_carton(df_sel_cart_sku, name_column)


def forming_df_for_tfidf(df_in):
    #функция преобразует столбец в cargotype в формат string, и для каждого sku формирует строку из всех его карготипов
    df = df_in.copy(deep=True)
    df['cargotype'] = df['cargotype'].astype('str')
    df_for_tfidf = df[['sku', 'cargotype']]
    df_for_tfidf['cargotype'] = df_for_tfidf['cargotype'] + ' '
    df_for_tfidf = df_for_tfidf.groupby('sku')['cargotype'].agg('sum').reset_index(drop=False)
    return df_for_tfidf


def get_tfidf_dataframe(df, path_to_save_vect):
    #функция производит векторизацию признаков при помощи tfidf и сохраняет обученный tfidf
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cargotype'])
    dict_for_columns = {i[1]: i[0] for i in tfidf.vocabulary_.items()}
    new_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix).rename(columns=dict_for_columns)
    new_df.insert(0, 'sku', df['sku'])
    pickle.dump(tfidf, open(path_to_save_vect, "wb"))
    return new_df


def save_sparse_csr(filename: str, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def score_function(y_true, y_pred):
    metrick_list = []
    for i, y_p in enumerate(y_pred):
        metrick_list.append(1 if y_true[i] in y_p else 0)
    return mean(metrick_list)


def x_train_predict_fias_n(x_train, x_test, y_test, k, n):
    d = len(x_train.columns)
    nb = x_train.shape[0]
    np.random.seed(123)
    xb = np.ascontiguousarray (x_train.values).astype('float32')
    xq_x_test = np.ascontiguousarray(x_test.values).astype('float32')
    nlist = 1
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained
    index.add(xb)
    x_test_n = x_test.iloc[:n]
    xq_x_test_n = xq_x_test[:n]
    y_test_n = y_test[:n].reset_index(drop=True)
    D, I = index.search(xq_x_test_n, k)
    predicted_list = []
    for candidates in I:
        predicted_list.append([id_base_dict[candidate] for candidate in candidates if candidate != -1])
    return score_function(y_test_n, predicted_list)


def x_get_index(x_train):
    d = len(x_train.columns)
    nb = x_train.shape[0]
    np.random.seed(123)
    xb = np.ascontiguousarray(x_train.values).astype('float32')
    nlist = 1
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained
    index.add(xb)
    return index


start_logger()
with FillingSquaresBar(' Прогресс выполнения', max=20) as bar:
    logger = logging.getLogger()
    logger.info(' Старт скрипта!')
    df_data = pd.read_csv(PATH_DATA)
    df_sku = pd.read_csv(PATH_SKU, index_col=0)
    df_sku_cargotypes = pd.read_csv(PATH_SKU_CARGOTYPES)
    df_carton = pd.read_csv(PATH_CARTON)
    bar.next()
    # этап предобработки данных, обучения масштабизаторов, векторизатора и создания датасетов для обучения моделей
    df_sku = cleaning_sku(df_sku)
    df_sku['volume'] = df_sku['a'] * df_sku['b'] * df_sku['c']
    df_sku['no_size'] = ((df_sku['a'].values == 0) &
                         (df_sku['b'].values == 0) &
                         (df_sku['c'].values == 0)).astype('int')
    bar.next()
    df_data = df_data[df_data['sku'].isin(df_sku['sku'])]
    df_data = df_data[df_data['goods_wght'] != 0]
    bar.next()
    df_cleaned = cleaning_data(df_data, df_carton, 'selected_carton')
    df_cleaned['selected_carton'].value_counts()
    df_cleaned = df_cleaned[df_cleaned['selected_carton'] != 'YMB']
    bar.next()
    df_rec = cleaning_data(df_data, df_carton, 'recommended_cartontype')
    df_rec = df_rec.merge(df_cleaned, how='inner', on='sku')
    df_tfidf = forming_df_for_tfidf(df_sku_cargotypes)
    df_tfidf = get_tfidf_dataframe(df_tfidf, PATH_TO_SAVE_TFIDF_VECTORIZER)
    bar.next()
    df_sku = df_sku[['sku', 'a', 'b', 'c', 'volume', 'no_size']].merge(df_tfidf, how='left', on='sku')
    df_sku['no_cargotype'] = (~df_sku['sku'].isin(df_tfidf['sku'])).astype('int')
    bar.next()
    df_sku = df_sku.fillna(0)
    df_for_classific = df_cleaned.merge(df_sku, how='left', on='sku')
    df_sku_wght = df_data.groupby(['sku', 'goods_wght'])['goods_wght'].agg(['count']).reset_index(drop=False)
    df_count_wght = df_sku_wght.groupby('sku')['goods_wght'].agg(['count']).reset_index(drop=False)
    df_sku_wght_bad = df_sku_wght[df_sku_wght['sku'].isin(df_count_wght[df_count_wght['count']>1]['sku'])]
    df_sku_wght_bad =\
        df_sku_wght_bad.groupby('sku')['goods_wght'].agg('mean').reset_index(drop=False).\
        rename(columns={'mean': 'goods_wght'})
    bar.next()
    df_sku_wght = df_sku_wght[df_sku_wght['sku'].isin(df_count_wght[df_count_wght['count']==1]['sku'])].drop('count', axis=1)
    df_sku_wght = pd.concat((df_sku_wght, df_sku_wght_bad), axis=0)
    df_for_classific = df_for_classific.merge(df_sku_wght, how='left', on='sku')
    df_for_classific['specific_weight'] = df_for_classific['goods_wght'] / df_for_classific['volume']
    df_for_classific['specific_weight'] = df_for_classific['specific_weight'].fillna(0)
    bar.next()
    df_for_classific.loc[df_for_classific['specific_weight']==np.inf, 'specific_weight'] = sys.maxsize
    scaler_for_classif = RobustScaler()
    df_for_classific[['a', 'b', 'c', 'volume', 'goods_wght', 'specific_weight']] = \
        scaler_for_classif.fit_transform(df_for_classific[['a', 'b', 'c', 'volume', 'goods_wght', 'specific_weight']])
    joblib.dump(scaler_for_classif, PATH_TO_SAVE_SCALER_CLASIFIC, compress=True)
    logger.info(' Обучен и выгружен масштабизатор для классификации!')
    bar.next()
    df_for_classific = df_for_classific.drop('sku', axis=1)
    df_for_classific = df_for_classific.rename(columns={'selected_carton': 'target'})
    df_rec = df_rec.drop('sku', axis=1)
    df_for_classific.to_csv(PATH_TO_SAVE_DF_FOR_CLASSIFIC, index=False)
    logger.info(' Выгружен датасет для классификации!')
    bar.next()
    df_rec.to_csv(PATH_TO_SAVE_REC_PRED, index=False)
    logger.info(' Выгружены предсказания для обучения моделей!')
    matrix_for_cluster = df_sku.copy(deep=True)
    matrix_for_cluster.drop(['sku'], axis=1, inplace=True)
    bar.next()
    scaler_for_cluster = RobustScaler()
    matrix_for_cluster[['a', 'b', 'c', 'volume']] =\
        scaler_for_cluster.fit_transform(matrix_for_cluster[['a', 'b', 'c', 'volume']])
    joblib.dump(scaler_for_cluster, PATH_TO_SAVE_SCALER_FOR_CLUSTER, compress=True)
    logger.info(' Обучен и выгружен масштабизатор для кластеризации!')
    bar.next()
    save_sparse_csr(PATH_TO_SAVE_SPARSE_FOR_CLUSTER, csr_matrix(matrix_for_cluster))
    logger.info(' Выгружена матрица для кластеризации!')
    bar.next()
    matrix_for_cluster_to_classif = df_cleaned.merge(df_sku, how='left', on='sku')
    matrix_for_cluster_to_classif[['a', 'b', 'c', 'volume']] =\
        (scaler_for_cluster.transform(matrix_for_cluster_to_classif[['a', 'b', 'c', 'volume']]))
    matrix_for_cluster_to_classif.drop(['sku', 'selected_carton'], axis=1, inplace=True)
    save_sparse_csr(PATH_TO_SAVE_SPARSE_FOR_CLUSTER_TO_CLASSIF, csr_matrix(matrix_for_cluster_to_classif))
    logger.info(' Выгружена матрица с кластерами для классификации!')
    bar.next()
    kmeans = KMeans(n_clusters=7, random_state=42).fit(matrix_for_cluster)
    pickle.dump(kmeans, open(PATH_TO_SAVE_CLUSTERING_MODEL, "wb"))
    logger.info(' Обучен и выгружена модель кластеризации!')
    bar.next()
    labels_for_df = kmeans.predict(matrix_for_cluster_to_classif)
    df_for_classific['cluster'] = labels_for_df
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(df_for_classific['cluster'].values.reshape(-1, 1))
    pickle.dump(ohe, open(PATH_TO_SAVE_ENCODER, "wb"))
    logger.info(' Обучен и выгружен энкодер!')
    bar.next()
    columns_for_ohe = ['cluster_' + str(i) for i in ohe.categories_[0]]
    df_for_classific[columns_for_ohe] = ohe.transform(df_for_classific['cluster'].values.reshape(-1, 1)).toarray()
    df_for_classific.drop('cluster', axis=1, inplace=True)
    X = df_for_classific.drop('target', axis=1)
    y = df_for_classific['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.3, stratify=y)
    x_train = x_train.copy(deep=True)
    x_test = x_test.copy(deep=True)
    id_base_dict = dict(y_train.reset_index(drop=True))
    bar.next()
    score_all = x_train_predict_fias_n(x_train, x_test, y_test, 3, 10_000)
    column_drop = []
    top_score = score_all
    for col in x_train.columns:
        column_drop.append(col)
        x_train_new = x_train.copy(deep=True)
        x_test_new = x_test.copy(deep=True)
        x_train_new = x_train_new.drop(column_drop, axis=1)
        x_test_new = x_test_new.drop(column_drop, axis=1)
        new_score = x_train_predict_fias_n(x_train_new, x_test_new, y_test, 3, 10_000)
        if top_score > new_score:
            column_drop.remove(col)
        else:
            top_score = new_score
    with open(PATH_TO_SAVE_COLUMNS_TO_DROP, "wb") as file:
        pickle.dump(column_drop, file)
    logger.info(' Выгружен список колонок для сброса!')
    bar.next()
    df_for_classific['target'].to_csv(PATH_TO_SAVE_COLUMNS_TARGET, index=False)
    logger.info(' Выгружен целевой признак для матчинга!')
    bar.next()
    df_for_classific.drop(column_drop, axis=1, inplace=True)
    df_for_classific.drop(['target'], axis=1, inplace=True)
    ind = x_get_index(df_for_classific)
    write_index(ind, PATH_TO_SAVE_FAISS)
    logger.info(' Обучен и выгружен индекс FAISS!')
    bar.next()

