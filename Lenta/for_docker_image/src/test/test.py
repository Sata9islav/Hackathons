import requests


data_1 = {"data": [{"store": "c81e728d9d4c2f636f067f89cc14862c",
                    "sku": "c7b711619071c92bef604c7ad68380dd",
                    "forecast_date": "2020-07-07"}]}

r_1 = requests.post("http://0.0.0.0:8001/forecast", json=data_1)
print(r_1.json())
