import joblib
import numpy as np
import os
import pandas as pd
import pickle
import sys

cwd = os.getcwd()
PATH_TO_SCALER_CLASIFIC = cwd + '/scalers/scaler_rb_for_clusific.bin'
PATH_TO_SCALER_FOR_CLUSTER = cwd + '/scalers/scaler_rb_for_cluster.bin'
PATH_TO_TFIDF_VECTORIZER = cwd + '/vectorizers/tfidf_vectorizer.bin'
PATH_TO_SAVE_ENCODER = cwd + '/scalers/encoder.bin'

# read models
with open(PATH_TO_SCALER_CLASIFIC, 'rb') as file:
    scaler_rb_for_clusific = joblib.load(file)

with open(PATH_TO_SCALER_FOR_CLUSTER, 'rb') as file:
    scaler_rb_for_cluster = joblib.load(file)

with open(PATH_TO_TFIDF_VECTORIZER, 'rb') as file:
    tfidf_vectorizer = joblib.load(file)

with open(PATH_TO_SAVE_ENCODER, 'rb') as file:
    encoder = pickle.load(file)


# read json
def read_data(lst_wth_dct):
    items_df = pd.DataFrame(lst_wth_dct)
    items_df = items_df.rename(columns={'type': 'cargotype'})
    items_df['cargotype'] = items_df['cargotype'].apply(lambda x: " ".join(x))
    items_df = items_df.loc[items_df.index.repeat(items_df['count'])].reset_index(drop=True)
    items_df.drop(['count'], axis=1, inplace=True)
    items_df.columns = ['sku', 'a', 'b', 'c', 'goods_wght', 'cargotype']
    items_df[['a', 'b', 'c', 'goods_wght']] = items_df[['a', 'b', 'c', 'goods_wght']].astype('float')
    return items_df


def forming_df_for_tfidf(df_in):
    df = df_in.copy(deep=True)
    df['cargotype'] = df['cargotype'].astype('str')
    df_for_tfidf = df[['sku', 'cargotype']]
    df_for_tfidf['cargotype'] = df_for_tfidf['cargotype'] + ' '
    df_for_tfidf = df_for_tfidf.groupby('sku')['cargotype'].agg('sum').reset_index(drop=False)
    return df_for_tfidf


def sorted_dementions_sku(df):
    df['list_demensions'] = df.apply(lambda x: sorted([x['a'], x['b'], x['c']]), axis=1)
    df['a'] = df.apply(lambda x: x['list_demensions'][0], axis=1)
    df['b'] = df.apply(lambda x: x['list_demensions'][1], axis=1)
    df['c'] = df.apply(lambda x: x['list_demensions'][2], axis=1)
    df.drop('list_demensions', axis=1, inplace=True)
    return df


def get_tfidf_dataframe(df, tfidf):
    tfidf_matrix = tfidf.transform(df['cargotype'])
    dict_for_columns = {i[1]: i[0] for i in tfidf.vocabulary_.items()}
    new_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix).rename(columns=dict_for_columns)
    new_df.insert(0, 'sku', df['sku'])
    # new_df = new_df.merge(df.drop('cargotype', axis=1), how='left', on='sku')
    new_df['no_cargotype'] = (df['cargotype'] == '').astype('int')
    return new_df


def get_df_for_cluster(df, tfidf, scaler):
    new_df = df.copy(deep=True)
    df_cargotype = get_tfidf_dataframe(new_df[['sku', 'cargotype']], tfidf)
    new_df = new_df[['sku', 'a', 'b', 'c']]
    new_df['volume'] = new_df['a'] * new_df['b'] * new_df['c']
    new_df = new_df.merge(df_cargotype.drop_duplicates(), how='left', on='sku')
    new_df['no_cargotype'] = (df['cargotype'] == '').astype('int')
    new_df['no_size'] = ((new_df['a'].values == 0) &
                         (new_df['b'].values == 0) &
                         (new_df['c'].values == 0)).astype('int')
    new_df[['a', 'b', 'c', 'volume']] = scaler.transform(new_df[['a', 'b', 'c', 'volume']])
    new_df.drop(['sku'], axis=1, inplace=True)
    return new_df


def get_weight_size_charact(df):
    new_df = df.copy(deep=True)
    new_df['specific_weight'] = new_df['goods_wght'] / new_df['volume']
    new_df['specific_weight'] = new_df['specific_weight'].fillna(0)
    new_df.loc[new_df['specific_weight'] == np.inf, 'specific_weight'] = sys.maxsize
    new_df = new_df.fillna(0)
    return new_df


def get_df_for_predict(df, tfidf, scaler, column_drop):
    new_df = df[['sku', 'a', 'b', 'c', 'cluster']]
    new_df['volume'] = new_df['a'] * new_df['b'] * new_df['c']
    new_df['no_size'] = ((df['a'].values == 0) &
                         (df['b'].values == 0) &
                         (df['c'].values == 0)).astype('int')
    new_df = sorted_dementions_sku(new_df)
    df_tfidf = get_tfidf_dataframe(df[['sku', 'cargotype']], tfidf)
    new_df = new_df.merge(df_tfidf, how='left', on='sku')
    new_df['goods_wght'] = df['goods_wght']
    new_df = get_weight_size_charact(new_df)
    new_df[['a', 'b', 'c', 'volume', 'goods_wght', 'specific_weight']] = scaler.transform(
        new_df[['a', 'b', 'c', 'volume', 'goods_wght', 'specific_weight']])
    columns_for_ohe = ['cluster_' + str(i) for i in encoder.categories_[0]]
    new_df[columns_for_ohe] = encoder.transform(new_df['cluster'].values.reshape(-1, 1)).toarray()
    new_df.drop('cluster', axis=1, inplace=True)
    new_df = new_df.drop(column_drop, axis=1)
    new_df = new_df.drop('sku', axis=1)
    return new_df
