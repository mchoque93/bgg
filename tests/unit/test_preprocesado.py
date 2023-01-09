from unittest.mock import patch

import pandas as pd
import pytest

from services.preprocesado.preprocesado import Preprocesado


class TestPreprocesado:
    @pytest.fixture(scope="class")
    def data_input(self):
        data_input = pd.DataFrame([(36, 5, 7, 0, 0, 48), (0, 0, 1, 0, 0, 1)],
                                  columns=['LD_one', 'LD_two', 'LD_three', 'LD_four', 'LD_five', 'LD_total_votes'])
        return data_input

    @patch("services.preprocesado.preprocesado.load_csv")
    def test_create_language_depende(self, mocker, data_input):
        mocker.return_value = data_input
        preprocesado = Preprocesado.define_preprocesado()
        df_lenguage_depence = preprocesado.create_language_depende()
        # THEN: Assert something
        assert all(df_lenguage_depence["L_A"].isin([0, 1]))
        assert all(df_lenguage_depence["L_A"]) == 0
        assert df_lenguage_depence["L_M"][0] == 0
        assert df_lenguage_depence["L_M"][1] == 1
        assert df_lenguage_depence["L_B"][0] == 1
        assert df_lenguage_depence["L_B"][1] == 0

    @patch("services.preprocesado.preprocesado.load_csv")
    def test_get_dummies(self, mocker):
        data_input = pd.DataFrame(
            [('Memory, Set Collection'), ('Hand Management, Investment, Market, Ownership')],
            columns=['mechanic'])
        mocker.return_value = data_input
        preprocesado = Preprocesado.define_preprocesado()
        dummies = preprocesado.get_dummies('mechanic')
        assert dummies['Memory'][0]==1 & dummies['Set Collection'][0]==1
        assert dummies['Hand Management'][1] == 1 & dummies['Investment'][1] == 1 & dummies['Market'][1] == 1 & dummies['Ownership'][1] == 1

    @patch("services.preprocesado.preprocesado.load_csv")
    def test_get_texto(self, mocker):
        data_input = pd.DataFrame(
            [("Political encourage use character 's authority manipulate societal activity policy .")],
            columns=['Theme definition'])
        mocker.return_value = data_input
        preprocesado = Preprocesado.define_preprocesado()
        preprocesado.drop_missings('Theme definition')
        preprocesado.processing_text()
        assert all(preprocesado.dataframe['Theme definition'].str.islower())
        assert not all(preprocesado.dataframe['Theme definition'].isin(['[(\.!?/\'&-,)]']))
