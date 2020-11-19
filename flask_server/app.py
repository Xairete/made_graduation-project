from flask import Flask, request, jsonify, send_file
from PIL import Image
from food_classifier import FoodClassifier

app = Flask(__name__)


FOOD_CLF = FoodClassifier()

@app.route('/predict', methods=['POST'])
def predict_router():
    files = request.files.to_dict(flat=False)
    
    answer_result = {}
    for file in files['images']:
        stream = file.stream
        image = Image.open(stream)
        pred_class = FOOD_CLF.predict_class(image)
        answer_result[file.filename] = pred_class
    return jsonify(answer_result)

