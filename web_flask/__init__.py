# Import model
from web_flask.model_evaluate import ModelEvaluate
model = ModelEvaluate()

from flask import Flask
from flask import request
from flask import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'i-will-never-give-you-up'

from web_flask import routes
app.run(host='127.0.0.1', port=5021, threaded=True, debug=False)
