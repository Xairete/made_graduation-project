import base64
from io import BytesIO
from typing import List

import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for
from flask.views import View
from igramscraper.exception import InstagramNotFoundException, InstagramAuthException
from igramscraper.instagram import Instagram
from PIL import Image
from sqlalchemy import create_engine

from config import Config
from helpers.helpers import PostStorage, ImageMeta, resize_im_bytes
from ml_engine.reco import get_b64_images, get_recommend
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FoodDetector, select_food
from ml_engine.food_selector import FoodSelector

app = Flask(__name__)


class AppContext(object):
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)
    food_selector = FoodSelector(Config.SELECTOR_FNAME)


CONTEXT = AppContext()
POST_STORAGE = PostStorage()
ENGINE=create_engine('sqlite:///df/meta.db')
RECO_DF = pd.read_sql_table(
    'reco',
    con=ENGINE,
    index_col=['index'],
)
REST_URL_DF = pd.read_sql_table(
    'rest_url',
    con=ENGINE,
    index_col=['restaurant_name']
)


class AddImages(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == "POST":
            image = request.files['file']
            image_bytes = image.stream.read()
            if len(image_bytes):
                caption = request.form['caption']
                image_b64 = base64.b64encode(
                    resize_im_bytes(image_bytes)).decode("ascii")
                POST_STORAGE.add(ImageMeta(image_bytes, image_b64, caption))
            return redirect(url_for('index'))

        if request.method == "GET":
            return render_template('add_image.html')


class PreprocessImages(View):
    methods = ['GET']

    def dispatch_request(self):
        images_meta = list(POST_STORAGE.images_meta.values())
        images = [Image.open(BytesIO(im.im_bytes)) for im in images_meta]
        if len(images):
            is_food_labels = CONTEXT.food_detector.predict(images)
            food_images_meta = select_food(images_meta, is_food_labels)
            if not len(food_images_meta):
                """
                Нет еды, удаляем
                """
                POST_STORAGE.clean()
                return redirect(url_for('index'))
            crop_images_meta = CONTEXT.food_selector.predict(food_images_meta)
            POST_STORAGE.clean()
            for im_meta in crop_images_meta:
                POST_STORAGE.add(im_meta)
        return redirect(url_for('index'))


class IndexView(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        im_dict = {key: (value.im_b64, value.caption)
                   for key, value in POST_STORAGE.images_meta.items()}
        return render_template('index.html', im_dict=im_dict)


class RemoveImage(View):
    methods = ['GET']

    def dispatch_request(self):
        rem_key = request.values.get("remove_image")
        rem_key = int(rem_key)
        POST_STORAGE.remove(rem_key)
        return redirect(url_for('index'))


class RecoView(View):
    methods = ['GET']

    def dispatch_request(self):
        meta_list = list(POST_STORAGE.images_meta.values())
        if len(meta_list):
            return self.recomendation_impl(meta_list)
        else:
            return redirect(url_for('index'))

    def recomendation_impl(self, meta_list):
        image_list = []
        for meta in meta_list:
            img_pic = Image.open(BytesIO(meta.im_bytes))
            image_list.append(img_pic)
        image_embeddings = CONTEXT.food_clf.predict(image_list)
        reco_rests = get_recommend(RECO_DF, image_embeddings)
        reco_dict = {}
        restraunt_url = REST_URL_DF.to_dict()['logo_url']
        for res_data in reco_rests:
            rest_dishes = []
            for dish_meta in res_data.rec_dishes[:5]: #TODO: сделать по нормальному
                rest_dishes.append(
                    {"url": dish_meta.dish_url, "name": dish_meta.dish_name, "score": dish_meta.score})
            rest_url=restraunt_url.get(res_data.rest_name,"")
            reco_dict[res_data.rest_name] = {
                "dishes": rest_dishes, "score": res_data.score, "rest_url":rest_url}
        return render_template('reco.html', reco_dict=reco_dict)


class InstagramParserView(View):
    methods = ['GET', 'POST']

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
            image_bytes = response.content
            """
            рескейлим base64 для рисования
            """
            image_b64 = base64.b64encode(
                resize_im_bytes(image_bytes)).decode("ascii")
            POST_STORAGE.add(ImageMeta(image_bytes, image_b64, media.caption))

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
            except InstagramAuthException:
                pass
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
