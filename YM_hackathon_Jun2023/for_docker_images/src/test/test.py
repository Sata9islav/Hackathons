import random
import os
import pandas as pd
import requests


cwd = os.getcwd()

data_2 = {"orderId": "unique_order_id",
 "items": [
    {"sku": "unique_sku_1", "count": 1, "size1": "5.1", "size2": "2.2", "size3": "5.3",
     "weight": "7.34", "type": ["210"]},
    {"sku": "unique_sku_2", "count": 3, "size1": "4", "size2": "5.23", "size3": "6.2",
     "weight": "7.45", "type": ["8", "90", "110"]},
    {"sku": "unique_sku_3", "count": 2, "size1": "11", "size2": "12.5", "size3": "13.3",
     "weight": "14.2", "type": ["150", "16"]}
   ]
}
# data_2 = {"orderId": "d48f3211c1ffccdc374f23139a9ab668",
#  "items": [
#     {"sku": "5f863de7185b639dc6a628704ed17484", "count": 1, "size1": "11.0", "size2": "6.0", "size3": "31.0",
#      "weight": "0.1", "type": ["290", "600", "610", "950", "970", "980"]},
#     {"sku": "af49bf330e2cf16e44f0be1bdfe337bd", "count": 17, "size1": "11.0", "size2": "6.0", "size3": "31.0",
#          "weight": "0.1", "type": ["290", "310", "610", "950", "970", "980"]}
#    ]
# }


PATH_TO_TEST_DATA_3 = cwd + '/data_for_test/data_for_test_3.csv'
data_3 = pd.read_csv(PATH_TO_TEST_DATA_3)
data_3[['goods_wght', 'a', 'b', 'c']] = data_3[['goods_wght', 'a', 'b', 'c']].astype('str')
data_3['count'] = random.randint(1, 2)
data_3['cargotype'] = data_3['cargotype'].apply(lambda x: x.split())
data_3.rename(columns={'a': 'size1', 'b': 'size2', 'c': 'size3', 'goods_wght': 'weight', 'cargotype': 'type'},
              inplace=True)
sku = []
for one_sku in data_3.sku.unique().tolist():
    full_one_product = data_3.query('sku == @one_sku')
    one_entry = full_one_product.to_dict('records')
    sku = sku + one_entry
data_3 = {"orderId": "unique_order_id",
          "items": sku}


r_2 = requests.post("http://0.0.0.0:8001/pack", json=data_2)
print(r_2.json())
# r_3 = requests.get("http://0.0.0.0:8001/pack", json=data_3)
# print(r_3.json())

