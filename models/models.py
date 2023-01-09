# features used in the model
import time

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV

from services.preprocesado.preprocesado import Preprocesado




class Modelo(Preprocesado):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.kfold = None

    def split_data(self):
        X = self.dataframe.loc[:, self.dataframe.columns != 'rating_average']
        y = self.dataframe['rating_average']

        # create train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            shuffle=True,
                                                            random_state=42)
        # have the model be split into 10 folds
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.kfold = kfold

    def clf_scores(self, clf):
        training_score = clf['gs'].cv_results_['mean_train_score'][clf['gs'].best_index_]
        training_std = clf['gs'].cv_results_['std_train_score'][clf['gs'].best_index_]
        print("Training Score = {:.3f}".format(training_score * -1)
              + " ({:.3f})".format(training_std))

        validation_score = clf['gs'].best_score_
        test_std = clf['gs'].cv_results_['std_test_score'][clf['gs'].best_index_]
        print("Validation Score = {:.3f}".format(validation_score * -1)
              + " ({:.3f})".format(test_std))

        train = "{:.3f}".format(training_score * -1) + " ({:.3f})".format(training_std)
        test = "{:.3f}".format(validation_score * -1) + " ({:.3f})".format(test_std)

        return [train, test]

    def clf_rsm(self, model):
        predict_y = model.predict(self.X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, predict_y))
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, predict_y))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test, predict_y)))

    def clf_features(self, pipe, model_type):
        self.split_data()
        models = []
        t0 = time.time()

        clf=pipe.fit(self.X_train, self.y_train)

        model = model_type
        models.append(clf)

        print(model.upper())
        self.clf_scores(clf)

        print("Métricas train")
        self.clf_rsm(clf)
        print("Métricas test")
        self.clf_rsm(clf)

        t1 = time.time()
        timeit = t1 - t0
        print('Run Time Mins: ', round((timeit / 60), 2))

        return models

