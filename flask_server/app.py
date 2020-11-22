from flask import Flask, request, jsonify, send_file
from PIL import Image
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FoodDetector, FOOD_LABEL
from flask.views import View
from config import Config

app = Flask(__name__)



def select_food(images, names, labels):
    food_images = []
    food_names = []

    for i in range(len(labels)):

        if labels[i] == FOOD_LABEL:
            food_names.append(names[i])
            food_images.append(images[i])

    return food_images, food_names
class MyView(View):
    """
    Вью, получает картинки, возвращает результат
    """
    methods = ['POST']
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)

    def dispatch_request(self):
        if request.method == 'POST':
            files = request.files.to_dict(flat=False)
            answer_result = {}
            images = []
            names = []
            for file in files['images']:
                stream = file.stream
                image = Image.open(stream)
                images.append(image)
                names.append(file.filename)

            is_food_labels = self.food_detector.predict(images)
            food_images, food_names = select_food(images, names, is_food_labels)

            food_labels = self.food_clf.predict(food_images)
            for name, label in zip(food_names, food_labels):
                answer_result[name] = int(label)
            return jsonify(answer_result)

app.add_url_rule('/predict', view_func=MyView.as_view('myview'))
