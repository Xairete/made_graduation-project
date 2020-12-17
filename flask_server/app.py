import base64
from io import BytesIO
from ml_engine.nlp_module import filter_positive_comments

from flask import Flask, redirect, render_template, request, url_for
from flask.views import View
from PIL import Image

from helpers.helpers import ImageMeta
from helpers.states import CONTEXT, POST_STORAGE
from ml_engine.food_detector import select_food
from views.parse_view import InstagramParserView
from views.reco_view import RecoView

app = Flask(__name__)


class AddImages(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        if request.method == "POST":
            image = request.files['file']
            image_bytes = image.stream.read()
            if len(image_bytes):
                caption = request.form['caption']
                image_b64 = base64.b64encode(
                    image_bytes).decode("ascii")
                POST_STORAGE.add(ImageMeta(image_bytes, image_b64, caption))
            return redirect(url_for('index'))

        if request.method == "GET":
            return render_template('add_image.html')


class PreprocessImages(View):
    methods = ['GET']

    def dispatch_request(self):
        images_meta = list(POST_STORAGE.images_meta.values())
        images_meta = filter_positive_comments(images_meta)
        if len(images_meta):
            images = [Image.open(BytesIO(im.im_bytes)) for im in images_meta]
            is_food_labels = CONTEXT.food_detector.predict(images)
            food_images_meta = select_food(images_meta, is_food_labels)
            if not len(food_images_meta):
                """
                Нет еды, удаляем
                """
                POST_STORAGE.clean()
                return redirect(url_for('index'))
            crop_images_meta = CONTEXT.food_selector.predict(food_images_meta)
            images = [Image.open(BytesIO(im.im_bytes)) for im in crop_images_meta]
            is_food_labels = CONTEXT.food_detector.predict(images)
            food_images_meta = select_food(crop_images_meta, is_food_labels)
            POST_STORAGE.clean()
            for im_meta in food_images_meta:
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

class Clean(View):
    methods = ['GET']

    def dispatch_request(self):
        POST_STORAGE.clean()
        return redirect(url_for('index'))


app.add_url_rule('/remove', view_func=RemoveImage.as_view('remove_image'))
app.add_url_rule('/recommend', view_func=RecoView.as_view('recommend'))
app.add_url_rule('/', view_func=IndexView.as_view('index'))
app.add_url_rule('/add_image', view_func=AddImages.as_view('add_image'))
app.add_url_rule(
    '/inst_parse', view_func=InstagramParserView.as_view('inst_parse'))
app.add_url_rule(
    '/preprocess', view_func=PreprocessImages.as_view('preprocess'))
app.add_url_rule(
    '/clean', view_func=Clean.as_view('clean'))
app.run(threaded=False)
