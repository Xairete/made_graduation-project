import base64
from helpers.helpers import ImageMeta
import os
import sys

import requests
from flask import render_template
from flask.views import View
from helpers.states import POST_STORAGE
from flask import redirect, render_template, request, url_for
from igramscraper.instagram import Instagram
from igramscraper.exception import (InstagramAuthException,
                                    InstagramNotFoundException)

sys.path.append("..")
MIN_SCORE = 4.5

template_dir = os.path.abspath('../temlates')


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
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            POST_STORAGE.add(ImageMeta(image_bytes, image_b64, media.caption))

    def dispatch_request(self):
        if request.method == "POST":

            account_name = request.form['instagram_url']
            num_parse = int(request.form['num_images'])
            try:
                instagram = Instagram()
                medias = instagram.get_medias(account_name, num_parse)
                self.parse_medias(medias)
            except InstagramNotFoundException:
                pass
                # TODO: add logging
            except InstagramAuthException:
                pass
            return redirect(url_for('index'))

        return render_template('instagram_parse.html')
