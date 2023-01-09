from flask_marshmallow import Schema
from marshmallow import fields


class DataFrameSchema(Schema):
    name = fields.String(required=False)
    min_players = fields.Integer(required=False)
    max_players = fields.Integer(required=False)
    playing_time = fields.Integer(required=False)
    min_age = fields.Integer(required=False)
    families = fields.String(required=False)
    mechanic = fields.String(required=False)
    domain = fields.String(required=False)
    genre_mechanic = fields.String(required=False)
    theme_definition = fields.String(required=False)
    component = fields.String(required=False)
    L_B = fields.Integer(required=False)
    L_M = fields.Integer(required=False)
    L_A = fields.Integer(required=False)
    desv_jg = fields.Integer(required=False)
    weight_average = fields.Integer(required=False)















