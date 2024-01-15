from flask import Flask, request, jsonify
from classify import classify_image
from recomendation import Recommender
import os


app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return "Hello Flask Server"


@app.route('/recomendation', methods=['POST'])
def get_recomendation():
    request_json = request.json
    nutrition_status = request_json['nutrition_status']
    food_type = request_json['food_type']
    result = Recommender.recomend(nutrition_status,  food_type)
    data_dic = result.to_dict()
    return jsonify({'food_list': data_dic})


@app.route('/prediction', methods=['POST'])
def classify():
    request_json = request.json
    url = request_json['image_url']
    class_dict = classify_image(url)
    return jsonify({'food_data': class_dict})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
