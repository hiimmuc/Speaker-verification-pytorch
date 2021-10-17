from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

db = {"nam": {'age': 20, 'score': 100},
      "hoa": {'age': 21, 'score': 90},
      "hai": {'age': 22, 'score': 80}}


class HelloWorld(Resource):
    def get(self, name='', num=0):
        return db[name]

    def post(self):
        return {'data': 'posted data'}


api.add_resource(HelloWorld, '/<string:name>')
#  if end point is hello
# api.add_resource(HelloWorld, '/hello')
if __name__ == '__main__':
    app.run(debug=True)
