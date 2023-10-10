import argparse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import json
import numpy as np
import logging
from pydantic import BaseModel
from typing import Dict, List
from preprocessing import get_preprocess_data
from predictions import prediction

app = FastAPI()


class Forecast(BaseModel):
    data: List


app = FastAPI()

_logger = logging.getLogger(__name__)


@app.post("/forecast")
def main(request: Forecast) -> dict:
    _logger.info(f'successfully start')
    items = request.data[0]
    data = get_preprocess_data.main([items['store']], [items['sku']],
                                    items['forecast_date'], 14)
    _logger.info(f'preprocessing complete')
    predict = (prediction.get_predictions(data))[['date', 'pr_sku_id', 'st_id', 'prediction_label']]
    _logger.info(f'prediction complete')
    predict = predict.to_dict(orient='records')
    _logger.info(f'prediction formated for return')
    return {"data": predict,
            "status": "ok"}


def setup_logging():
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_handler = logging.StreamHandler()
    app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    app_handler.setFormatter(app_formatter)
    app_logger.addHandler(app_handler)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8001, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    # parser.add_argument("--debug", action="store_true", dest="debug")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
    uvicorn.run(app, host="0.0.0.0", port=8001, debug=True)