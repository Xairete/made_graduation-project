import base64
from io import BytesIO
from typing import List
import pandas as pd

import requests
from flask import Flask, redirect, render_template, request, url_for
from flask.views import View
from igramscraper.exception import InstagramNotFoundException
from igramscraper.instagram import Instagram
from PIL import Image
from helpers.reco import get_recommend, get_b64_images

from config import Config
from helpers.helpers import FakeDb, ImageMeta
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FoodDetector, select_food
from ml_engine.food_selector import FoodSelector

app = Flask(__name__)


class AppContext(object):
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)
    food_selector = FoodSelector(Config.SELECTOR_FNAME)


CONTEXT = AppContext()
DB = FakeDb()
RECO_DF = pd.read_parquet("df_all_embs.parquet.gzip")
RECO_DF['rest_name']=RECO_DF['image_path'].map(lambda x :x.split('/')[2])


class AddImages(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == "POST":
            image = request.files['file']
            image_bytes = image.stream.read()
            image_bytes = InstagramParserView.resize(image_bytes)
            if len(image_bytes):
                caption = request.form['caption']
                image_b64 = base64.b64encode(image_bytes).decode("ascii")
                DB.add(ImageMeta(image_bytes, image_b64, caption))
            return redirect(url_for('index'))

        if request.method == "GET":
            return render_template('add_image.html')


class PreprocessImages(View):
    methods = ['GET']

    def dispatch_request(self):
        images_meta = list(DB.images_meta.values())
        images = [Image.open(BytesIO(im.im_bytes)) for im in images_meta]
        if len(images):
            is_food_labels = CONTEXT.food_detector.predict(images)
            food_images_meta = select_food(images_meta, is_food_labels)
            crop_images_meta = CONTEXT.food_selector.predict(food_images_meta)
            DB.clean()
            for im_meta in crop_images_meta:
                DB.add(im_meta)
        return redirect(url_for('index'))


class IndexView(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        im_dict = {key: (value.im_b64, value.caption)
                   for key, value in DB.images_meta.items()}
        return render_template('index.html', im_dict=im_dict)


class RemoveImage(View):
    methods = ['GET']

    def dispatch_request(self):
        rem_key = request.values.get("remove_image")
        rem_key = int(rem_key)
        DB.remove(rem_key)
        return redirect(url_for('index'))


class RecoView(View):
    methods = ['GET']

    def dispatch_request(self):
        meta_list = list(DB.images_meta.values())
        image_list = []
        for meta in meta_list:
            img_pic = Image.open(BytesIO(meta.im_bytes))
            image_list.append(img_pic)
        image_embeddings = CONTEXT.food_clf.predict(image_list)
        result = get_recommend(RECO_DF, image_embeddings)
        b64_images = get_b64_images(result)
        # return redirect(url_for('index'))
        return render_template('reco.html', b64_images=b64_images)


class InstagramParserView(View):
    methods = ['GET', 'POST']

    @staticmethod
    def resize(image_bytes):
        im = Image.open(BytesIO(image_bytes))
        im = im.resize((256, 256))  # TODO : переделать
        img_byte_arr = BytesIO()
        im.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    @staticmethod
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
            image_bytes = InstagramParserView.resize(response.content)
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            DB.add(ImageMeta(image_bytes, image_b64, media.caption))

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


app.add_url_rule('/remove', view_func=RemoveImage.as_view('remove_image'))
app.add_url_rule('/recommend', view_func=RecoView.as_view('recommend'))
app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/add_image', view_func=AddImages.as_view('add_image'))
app.add_url_rule(
    '/inst_parse', view_func=InstagramParserView.as_view('inst_parse'))
app.add_url_rule(
    '/preprocess', view_func=PreprocessImages.as_view('preprocess'))
