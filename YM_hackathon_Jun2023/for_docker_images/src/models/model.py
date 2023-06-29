import numpy as np
import os
import pandas as pd
import pickle
from faiss import read_index

cwd = os.getcwd()
PATH_TO_CLUSTERING_MODEL = cwd + '/models/clustering_model.pkl'
PATH_TO_FAISS = cwd + "/models/fiass_index_with_drop_columns.index"

PATH_TO_COLUMNS_TARGET = cwd + '/models/target.csv'
PATH_TO_COLUMNS_TO_DROP = cwd + "/models/columns_to_drop"


with open(PATH_TO_CLUSTERING_MODEL, 'rb') as file:
    kmeans = pickle.load(file)

with open(PATH_TO_COLUMNS_TO_DROP, 'rb') as file:
    column_for_drop = pickle.load(file)

target = (pd.read_csv(PATH_TO_COLUMNS_TARGET))['target']
id_base_dict = dict(target.reset_index(drop=True))
index = read_index(PATH_TO_FAISS)


def get_fit_predict_list_n(index, x_test, k=20, n=3, one_row=True):
    xq_x_test = x_test.values.astype('float32')
    dem, ind = index.search(xq_x_test, k)
    predicted_list = []
    for candidates in ind:
        predict = []
        for candidate in candidates:
            if id_base_dict[candidate] not in predict:
                predict.append(id_base_dict[candidate])
        predicted_list.append(predict[:n])
    if one_row:
        return list(np.array(predicted_list).reshape(-1, ))
    return predicted_list


def get_bool_list_ib_list(list1, list2):
    res = False
    for carton in list1:
        if carton in list2:
            res = True
    return res
