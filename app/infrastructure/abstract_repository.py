from abc import abstractmethod


class AbstractRepository:

    @abstractmethod
    def load_model(self):
        raise NotImplementedError
