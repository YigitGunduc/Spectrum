from waitress import serve
from WebApp.routes import *

port = 5000

if __name__ == '__main__':
    serve(app=app, port=port)
