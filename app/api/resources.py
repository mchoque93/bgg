import json

import numpy as np
from apiflask import APIBlueprint
from flask import Response

from app.api.scheme import DataFrameSchema
from app.infrastructure.local_repository import LocalRepository
from app.service.dict_to_dataframe import dict_to_dataframe
from app.service.preprocesado_predict import preprocesado_predict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

bgg_v1_0_bp = APIBlueprint(
    "bgg_v1_0_bp", __name__, url_prefix="/bgg"
)

df_schema = DataFrameSchema()
local_repository = LocalRepository()


@bgg_v1_0_bp.post("/predict_lr_model")
@bgg_v1_0_bp.input(schema=df_schema)
def predict_rating(data):
    """
    :return:
    """
    lr_model = local_repository.load_model_lr()
    df_data = dict_to_dataframe(data)
    preprocesado_df_data = preprocesado_predict(df_data)
    y_predicted = lr_model.predict(preprocesado_df_data)
    return Response(json.dumps({'output': y_predicted}, cls=NumpyEncoder), status=200, mimetype="application/json")
