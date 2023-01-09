import collections
from itertools import chain

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from services.loaders.load_csv import load_csv
from services.preprocesado.preprocesado import Preprocesado


class LDAModel(Preprocesado):
    def __init__(self, dataframe, field, components):
        super().__init__(dataframe)
        self.components = components
        self.drop_missings(field)
        self.processing_text()

    @classmethod
    def define_preprocesado(cls, field, components):
        return cls(load_csv(), field, components)


    def vectorized_model(self):
        model = CountVectorizer(stop_words='english')
        X = model.fit_transform(self.dataframe['Theme definition'])
        return X

    def lda_model(self):
        # Build LDA Model
        lda_model = LatentDirichletAllocation(n_components=self.components, max_iter=100, learning_method='batch', random_state=100,
                                              evaluate_every=-1, n_jobs=-1)
        lda_output = lda_model.fit_transform(self.vectorized_model())
        return lda_model, lda_output

    def get_dominant_topics(self, lda_model, lda_output):
        # column names
        topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
        # index names
        docnames = self.dataframe["name"]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        self.dataframe['dominant_topic'] = dominant_topic
        return self.dataframe

    def train_lda(self):
        self.vectorized_model()
        lda_model, lda_output = self.lda_model()
        self.get_dominant_topics(lda_model, lda_output)
        self.get_dummies('dominant_topic', 'T', 'int')





