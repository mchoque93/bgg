import joblib

from app.infrastructure.abstract_repository import AbstractRepository


class LocalRepository(AbstractRepository):
    def load_model_lr(self):
        loaded_model = joblib.load("lr.joblib")
        return loaded_model