from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from services.preprocesado.preprocesado import Preprocesado

if TYPE_CHECKING:
    from pandas import DataFrame


class LDATransformer(BaseEstimator, TransformerMixin, Preprocesado):
    def __init__(self, components, field):
        self.components = components
        self.field = field
        self.lda_modelo = None
        self.lda_output = None
        self.dataframe = None
        self.vectorized_modelo = None

    def vectorized_model(self, X: "DataFrame"):
        self.vectorized_modelo = CountVectorizer(stop_words='english')
        X = self.vectorized_modelo.fit_transform(X['Theme definition'])
        return X

    def lda_model(self, X: "DataFrame"):
        # Build LDA Model
        lda_model = LatentDirichletAllocation(n_components=self.components, max_iter=100, learning_method='batch',
                                              random_state=100,
                                              evaluate_every=-1, n_jobs=-1)
        lda_output = lda_model.fit_transform(self.vectorized_model(X))
        return lda_model, lda_output

    def get_dominant_topics(self, X: "DataFrame", lda_model, lda_output):
        # column names
        topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
        # index names
        docnames = X["name"]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        X['dominant_topic'] = dominant_topic
        return X

    def fit(self, X: "DataFrame", y=None):
        lda_model, lda_output = self.lda_model(X)
        self.lda_modelo = lda_model
        self.lda_output = lda_output
        return self

    def transform(self, X, y=None):
        self.dataframe = X
        lda_output = self.lda_modelo.transform(self.vectorized_modelo.transform(X['Theme definition']))
        self.dataframe = self.get_dominant_topics(self.dataframe, self.lda_modelo, lda_output)
        self.get_dummies('dominant_topic', 'T', 'int')
        for component in range(self.components):
            if f'T{component}' not in self.dataframe:
                self.dataframe[f'T{component}'] = 0
            else:
                variable_auxiliar = self.dataframe[f'T{component}']
                self.dataframe = self.dataframe.drop([f'T{component}'], axis=1)
                self.dataframe[f'T{component}'] = variable_auxiliar
        return self.dataframe.drop(['name', 'Theme definition'], axis=1)
