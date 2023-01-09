from typing import TYPE_CHECKING

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

if TYPE_CHECKING:
    from pandas import DataFrame


class DummiesTransformer():
    def __init__(self, field, prefix="", type="string"):
        self.field = field
        self.prefix = prefix
        self.type = type
        self.mlb = None

    def get_dummies(self, X):
        print(self.field)
        if self.type == "string":
            X[self.field] = X[self.field].str.split(",")
        else:
            X[self.field] = X[self.field].astype(str)
        mlb = MultiLabelBinarizer()
        mlb.fit(X[self.field])
        return mlb

    def fit(self, X: "DataFrame", y=None):
        mlb = self.get_dummies(X)
        self.mlb = mlb
        return self

    def transform(self, X, y=None):
        dummies = pd.DataFrame(self.mlb.transform(X[self.field]), columns=[self.prefix + row.strip() for row in self.mlb.classes_],
                               index=X.index)
        X = pd.concat([X, dummies], axis=1, sort=False)
        return X
