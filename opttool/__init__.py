from flask import Flask
from schemas.serializer import configure_serializer


def create_app(config_type):

	app = Flask(__name__)
	app.config.from_object(config_type)

	configure_serializer(app)

	from views import api as api_blueprint
	app.register_blueprint(api_blueprint)

	return app