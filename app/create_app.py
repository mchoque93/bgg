from apiflask import APIFlask

from app.api.resources import bgg_v1_0_bp


#settings = os.getenv("APP_SETTINGS_MODULE", "config.DefaultConfig")


def create_app():
    app = APIFlask(__name__)
    #app.config.from_object(settings_module)

    # Inicializa las extensiones
    #with app.app_context():

    # Api(app, catch_all_404s=True)

    # Deshabilita el modo estricto de acabado de una URL con /
    # app.url_map.strict_slashes = False

    # Registra los blueprints
    app.register_blueprint(bgg_v1_0_bp)

    # Registra manejadores de errores personalizados
    # register_error_handlers(app)

    return app
