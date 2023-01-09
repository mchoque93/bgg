import re
from operator import ge, le

import pandas as pd
from numpy import dtype
from sklearn.preprocessing import MultiLabelBinarizer

from services.loaders.load_csv import load_csv


class Preprocesado:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @classmethod
    def define_preprocesado(cls):
        return cls(load_csv())

    def delete_outliers(self):
        conditions = [('num_rates', ge, 201), ('year_published', le, 2020), ('year_published', ge, 1976),
                      ('min_play_time', ge, 11), ('max_play_time', ge, 11), ('max_play_time', le, 479),
                      ('max_players', le, 14)]
        for column, operator, n in conditions:
            self.dataframe = self.dataframe[operator(self.dataframe[column], n)]
        return self.dataframe

    def create_language_depende(self):
        self.dataframe['L_B'] = (self.dataframe['LD_one'] + self.dataframe['LD_two']) / self.dataframe['LD_total_votes']
        self.dataframe['L_M'] = self.dataframe['LD_three'] / self.dataframe['LD_total_votes']
        self.dataframe['L_A'] = (self.dataframe['LD_four'] + self.dataframe['LD_five']) / self.dataframe[
            'LD_total_votes']

        self.dataframe['L_B'] = (self.dataframe['L_B'] >= 0.51)
        self.dataframe['L_B'] = self.dataframe['L_B'].astype(int)
        self.dataframe['L_M'] = (self.dataframe['L_M'] >= 0.51)
        self.dataframe['L_M'] = self.dataframe['L_M'].astype(int)
        self.dataframe['L_A'] = (self.dataframe['L_A'] >= 0.51)
        self.dataframe['L_A'] = self.dataframe['L_A'].astype(int)

        self.dataframe = self.dataframe.drop(['LD_one', 'LD_two', 'LD_three', 'LD_four', 'LD_five', 'LD_total_votes'], axis=1)

        return self.dataframe

    def create_flags(self):
        self.dataframe['flag_kickstarter'] = self.dataframe['families'].str.contains('Kickstarter', na=False)
        self.dataframe['flag_kickstarter'] = self.dataframe['flag_kickstarter'].fillna(0)
        self.dataframe["flag_kickstarter"] = self.dataframe["flag_kickstarter"].astype(int)
        return self.dataframe

    def get_dummies(self, field, prefix="", type="string"):
        if type == "string":
            self.dataframe[field] = self.dataframe[field].str.split(",")
        else:
            self.dataframe[field] = self.dataframe[field].astype(str)
        mlb = MultiLabelBinarizer()
        dummies = pd.DataFrame(mlb.fit_transform(self.dataframe[field]), columns=[prefix + row.strip() for row in mlb.classes_],
                               index=self.dataframe.index)
        self.dataframe = pd.concat([self.dataframe, dummies], axis=1, sort=False)
        return self.dataframe

    def escalabilidad_players(self):
        self.dataframe_std = pd.DataFrame({})
        self.dataframe_std[['1', '2', '3', '4', '5', '6']] = self.dataframe[['1', '2', '3', '4', '5', '6']].div(
            self.dataframe['num_players_votes'].values,
            axis=0)
        self.dataframe['desv_jg'] = self.dataframe_std.std(axis=1)
        self.dataframe['desv_jg'] = self.dataframe['desv_jg'].fillna(0)

        self.dataframe = self.dataframe.drop(['1', '2', '3', '4', '5', '6'], axis=1)

        return self.dataframe

    def processing_text(self):
        # Convert the titles to lowercase
        self.dataframe['Theme definition'] = self.dataframe['Theme definition'].map(
            lambda x: x.lower() if isinstance(x, str) and len(x) else 0)
        # Remove punctuation
        self.dataframe['Theme definition'] = self.dataframe['Theme definition'].map(
            lambda x: re.sub('[(\.!?/\'&-,)]', '', str(x)))
        return self.dataframe

    def fill_missings(self):
        return self.dataframe.fillna(value=0)

    def select_variables(self, features):
        return self.dataframe[features]

    def drop_missings(self, field):
        return self.dataframe.dropna(how='any', subset=[field])

    def fill_na(self):
        for field, field_type in list(zip(self.dataframe.dtypes.index, self.dataframe.dtypes.values)):
            if field_type == dtype('O') or field_type == dtype('str'):
                self.dataframe[field].fillna("", inplace=True)
            else:
                self.dataframe[field].fillna(0, inplace=True)
