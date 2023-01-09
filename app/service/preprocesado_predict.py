from services.preprocesado.preprocesado import Preprocesado


def preprocesado_predict(dataframe):
    preprocesado = Preprocesado(dataframe)

    preprocesado.create_flags()
    preprocesado.processing_text()


    return preprocesado.dataframe