import base64
from igramscraper.instagram import Instagram 
from multiprocessing import Process, Queue

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask.views import View
from PIL import Image

from config import Config
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FOOD_LABEL, FoodDetector
from helpers.helpers import FakeDb

app = Flask(__name__)


def select_food(images, names, labels):
    food_images = []
    food_names = []

    for i in range(len(labels)):

        if labels[i] == FOOD_LABEL:
            food_names.append(names[i])
            food_images.append(images[i])

    return food_images, food_names


class AppContext(object):
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)


CONTEXT = AppContext()
DB = FakeDb()


class PredictView(View):
    """
    Вью, получает картинки, возвращает результат
    """
    methods = ['POST']

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

            is_food_labels = CONTEXT.food_detector.predict(images)
            food_images, food_names = select_food(
                images, names, is_food_labels)

            food_labels = CONTEXT.food_clf.predict(food_images)
            for name, label in zip(food_names, food_labels):
                answer_result[name] = int(label)
            return jsonify(answer_result)


class AddImages(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == "POST":
            file = request.files['file']
            file_bytes = file.stream.read()
            if len(file_bytes):
                im = base64.b64encode(file_bytes)
                comment = request.form['comment']
                DB.add(im, comment)
        return render_template('index.html', images=DB.images)

class RemoveImage(View):
    methods = ['POST']

    def dispatch_request(self):
        rem_key = request.values.get("remove_image")
        rem_key = int(rem_key)
        DB.remove(rem_key)
        return redirect(url_for('add_image'))

class InstagramParserView(View):
    methods = ['GET', 'POST']
    def dispatch_request(self):
        if request.method == "POST":

            account_name = request.form['instagram_url']
            instagram = Instagram()
            medias = instagram.get_medias(account_name, 25)
            for m in medias:
                print(m.caption)

            print("inst_url={}".format(account_name))
        return render_template('instagram_parse.html')


app.add_url_rule('/predict', view_func=PredictView.as_view('predict_view'))
app.add_url_rule('/remove', view_func=RemoveImage.as_view('remove_image'))
app.add_url_rule('/', view_func=AddImages.as_view('add_image'))
app.add_url_rule('/inst_parse', view_func=InstagramParserView.as_view('inst_parse'))

