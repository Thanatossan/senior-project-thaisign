# flaskproject1/app/__init__.py
from flask import Flask
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'app/upload_video'
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8000)
    # app.run(host='0.0.0.0',port=8000)
from app import routes