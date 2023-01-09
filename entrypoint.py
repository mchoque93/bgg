import logging

import joblib
import numpy as np
import xgboost as xgb
from joblib import dump
from sklearn import feature_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from models.column_dropper_transformer import ColumnDropperTransformer
from models.dummies_transformer import DummiesTransformer
from models.lda_transformer import LDATransformer
from models.models import Modelo
from services.preprocesado.preprocesado import Preprocesado

logger = logging.getLogger(__name__)

drop_cols_pipeline = ['dominant_topic', 'mechanic', 'Domain', 'Genre (Mechanic)', 'Component',
                      'families', 'missing_value']


def preprocesado():
    logger.info("Comienza preprocesado")
    preprocesado = Preprocesado.define_preprocesado()
    preprocesado.delete_outliers()
    preprocesado.create_language_depende()
    preprocesado.escalabilidad_players()

    preprocesado.create_flags()

    preprocesado.drop_missings("Theme definition")
    preprocesado.processing_text()

    dataframe = preprocesado.dataframe

    drop_cols = ['type', 'year_published',
                 'min_play_time', 'max_play_time',
                 'expansions', 'designers', 'image_url', 'price', 'categories',
                 'Theme', 'Game System', 'num_players_votes', 'num_rates',
                 'bayes_average', 'std_dev', 'median', 'owned', 'trading', 'wanting',
                 'wishing', 'num_comments', 'num_weights', 'id', 'mechanics']

    dataframe = dataframe.drop(drop_cols, axis=1)
    return dataframe


def lr_model():
    logger.info("start training: LR model")
    dataframe = preprocesado()

    modelo = Modelo(dataframe)

    # Regresión lineal
    pipe_lr = Pipeline(
        [('simple_imputer', SimpleImputer(strategy='constant').set_output(transform="pandas")),
         ('dummy_mechanic', DummiesTransformer('mechanic')),
         ('dummy_component', DummiesTransformer('Component')),
         ('dummy_genre_mechanic', DummiesTransformer('Genre (Mechanic)')),
         ('dummy_domain', DummiesTransformer('Domain')),
         ('lda', LDATransformer(8, 'Theme definition')),
         ("columnDropper", ColumnDropperTransformer(drop_cols_pipeline)),
         ('scale', StandardScaler()),
         ('gs', GridSearchCV(LinearRegression(),
                             param_grid={'fit_intercept': [True, False]},
                             cv=modelo.kfold,
                             return_train_score=True,
                             refit=True))
         ])

    result = modelo.clf_features(pipe_lr, 'lr')

    print(result[0]['gs'].best_params_)
    print(result[0]['gs'].best_estimator_)

    dump(result[0], 'lr.joblib')


def rfr_model():
    logger.info("start training: RF model")
    dataframe = preprocesado()

    modelo = Modelo(dataframe)

    param_rfr = {'n_estimators': np.arange(1, 10).tolist(),
                 'max_features': ['sqrt', 'log2'],
                 'max_depth': np.arange(2, 6).tolist(),
                 'min_samples_split': np.arange(2, 10).tolist(),
                 'min_samples_leaf': np.arange(2, 10).tolist()
                 }

    pipe_rfr = Pipeline([('simple_imputer', SimpleImputer(strategy='constant').set_output(transform="pandas")),
                         ('dummy_mechanic', DummiesTransformer('mechanic')),
                         ('dummy_component', DummiesTransformer('Component')),
                         ('dummy_genre_mechanic', DummiesTransformer('Genre (Mechanic)')),
                         ('dummy_domain', DummiesTransformer('Domain')),
                         ('lda', LDATransformer(8, 'Theme definition')),
                         ("columnDropper", ColumnDropperTransformer(drop_cols_pipeline)),
                         ('scale', StandardScaler()),
                         ('gs', RandomizedSearchCV(RandomForestRegressor(),
                                                   param_distributions=param_rfr,
                                                   scoring='neg_mean_absolute_error',
                                                   cv=modelo.kfold,
                                                   random_state=42,
                                                   n_jobs=-1,
                                                   n_iter=1000,
                                                   return_train_score=True,
                                                   refit=True))
                         ])

    result = modelo.clf_features(pipe_rfr, 'rfr')
    print(result[0]['gs'].best_params_)
    print(result[0]['gs'].best_estimator_)

    dump(result[0], 'rfr.joblib')


def xgb_model():
    logger.info("start training: XGboost model")

    dataframe = preprocesado()

    modelo = Modelo(dataframe)

    param_xgb = {
        'n_estimators': [30, 60, 100, 150, 200],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'max_depth': [5, 7, 10, 12, 15],
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [1.1, 1.2, 1.3],
        'subsample': [0.7, 0.8, 0.9]
    }

    pipe_xgb = Pipeline([('simple_imputer', SimpleImputer(strategy='constant').set_output(transform="pandas")),
                         ('dummy_mechanic', DummiesTransformer('mechanic')),
                         ('dummy_component', DummiesTransformer('Component')),
                         ('dummy_genre_mechanic', DummiesTransformer('Genre (Mechanic)')),
                         ('dummy_domain', DummiesTransformer('Domain')),
                         ('lda', LDATransformer(8, 'Theme definition')),
                         ("columnDropper", ColumnDropperTransformer(drop_cols_pipeline)),
                         ('scale', StandardScaler()),
                         ('gs', RandomizedSearchCV(xgb.XGBRegressor(),
                                                   param_distributions=param_xgb,
                                                   scoring='neg_mean_absolute_error',
                                                   cv=2,
                                                   random_state=42,
                                                   n_jobs=-1,
                                                   n_iter=100,
                                                   return_train_score=True,
                                                   refit=True))
                         ])

    result = modelo.clf_features(pipe_xgb, 'xgboost')
    print(result[0]['gs'].best_params_)
    print(result[0]['gs'].best_estimator_)

    dump(result[0]['gs'].best_estimator_, 'xgboost.joblib')


def lr_model_selector():
    logger.info("start training: LR model selector")
    dataframe = preprocesado()

    modelo = Modelo(dataframe)


    selector = RFECV(estimator=DecisionTreeRegressor(), step = 1, cv = 5)

    pipe_lr = Pipeline(
        [('simple_imputer', SimpleImputer(strategy='constant').set_output(transform="pandas")),
         ('dummy_mechanic', DummiesTransformer('mechanic')),
         ('dummy_component', DummiesTransformer('Component')),
         ('dummy_genre_mechanic', DummiesTransformer('Genre (Mechanic)')),
         ('dummy_domain', DummiesTransformer('Domain')),
         ('lda', LDATransformer(8, 'Theme definition')),
         ("columnDropper", ColumnDropperTransformer(drop_cols_pipeline)),
         ('scale', StandardScaler()),
         ('selector', selector),
         ('gs', GridSearchCV(LinearRegression(),
                             param_grid={'fit_intercept': [True, False]},
                             cv=modelo.kfold,
                             return_train_score=True,
                             refit=True))
         ])

    result = modelo.clf_features(pipe_lr, 'lr')
    print(result[0]['gs'].best_params_)
    print(result[0]['gs'].best_estimator_)

    dump(result[0], 'lr_selector.joblib')



if __name__ == '__main__':
    #lr_model()
    #rfr_model()
    #xgb_model()
    lr_model_selector()
    loaded_model = joblib.load("lr_selector.joblib")


