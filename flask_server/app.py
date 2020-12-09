import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List

import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask.views import View
from igramscraper.exception import InstagramNotFoundException
from igramscraper.instagram import Instagram
from PIL import Image

from config import Config
from helpers.helpers import FakeDb
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FOOD_LABEL, FoodDetector

app = Flask(__name__)


class AppContext(object):
    pass
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)


CONTEXT = AppContext()
DB = FakeDb()


class AddImages(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == "POST":
            file = request.files['file']
            file_bytes = file.stream.read()
            if len(file_bytes):
                comment = request.form['comment']
                DB.add(file_bytes, comment)
            return redirect(url_for('index'))

        if request.method == "GET":
            return render_template('add_image.html')


class IndexView(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        im_dict = {key: (base64.b64encode(val[0]).decode(
            "ascii"), val[1]) for key, val in DB.images.items()}
        return render_template('index.html', im_dict=im_dict)


class RemoveImage(View):
    methods = ['POST']

    def dispatch_request(self):
        rem_key = request.values.get("remove_image")
        rem_key = int(rem_key)
        DB.remove(rem_key)
        return redirect(url_for('add_image'))


@ dataclass
class ImageMeta:
    im_bytes: bytes
    comments: str


def select_food(meta: List[ImageMeta], labels):
    result = []
    for i in range(len(labels)):
        if labels[i] == FOOD_LABEL:
            im_b64 = meta[i].im_bytes
            result((im_b64, meta[i].comments))


class InstagramParserView(View):
    methods = ['GET', 'POST']

    @ staticmethod
    def parse_medias(medias):

        for media in medias:
            image_url = None
            if not media.image_high_resolution_url is None:
                image_url = media.image_high_resolution_url
            elif not media.image_standard_resolution_url is None:
                image_url = media.image_standard_resolution_url
            elif not media.image_low_resolution_url is None:
                image_url = media.image_low_resolution_url
            if image_url is None:
                continue
            response = requests.get(image_url, stream=True)
            image_byte = response.content
            DB.add(image_byte, media.caption)

    def dispatch_request(self):
        if request.method == "POST":

            account_name = request.form['instagram_url']
            num_parse = int(request.form['num_images'])
            instagram = Instagram()
            try:
                medias = instagram.get_medias(account_name, num_parse)
                self.parse_medias(medias)
            except InstagramNotFoundException:
                pass
                # TODO: add logging
            return redirect(url_for('index'))

        return render_template('instagram_parse.html')


# app.add_url_rule('/predict', view_func=PredictView.as_view('predict_view'))
app.add_url_rule('/remove', view_func=RemoveImage.as_view('remove_image'))
app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/add_image', view_func=AddImages.as_view('add_image'))
app.add_url_rule('/inst_parse', view_func=InstagramParserView.as_view('inst_parse'))
