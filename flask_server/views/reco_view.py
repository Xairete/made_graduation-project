import heapq
import os
import random
import sys
from io import BytesIO

from flask import render_template
from flask.views import View
from helpers.states import CONTEXT, POST_STORAGE, RECO_DF, REST_URL_DF
from ml_engine.reco import get_recommend
from PIL import Image

sys.path.append("..")
MIN_SCORE = 4.5

template_dir = os.path.abspath('../temlates')


def remove_repeat(dish_list):
    name_to_dish = {}
    for dish in dish_list:
        name = dish.dish_name
        if name not in name_to_dish:
            name_to_dish[name] = dish
        elif name_to_dish[name].score > dish.score:
            name_to_dish[name] = dish

    return list(name_to_dish.values())


class RecoView(View):
    methods = ['GET']

    def dispatch_request(self):
        meta_list = list(POST_STORAGE.images_meta.values())
        if len(meta_list):
            reco_dict = self.recomendation_impl(meta_list)
        else:
            reco_dict = self.reco_cold_start()
        return render_template('reco.html', reco_dict=reco_dict, template_folder=template_dir)

    def reco_cold_start(self):
        high_score_df = RECO_DF[RECO_DF.score > MIN_SCORE]
        groups = high_score_df[['dish_name', 'url', 'restaurant_name']].groupby([
                                                                                'restaurant_name'])
        group_list = [group for group in groups]
        random.shuffle(group_list)

        reco_dict = {}
        restraunt_url = REST_URL_DF.to_dict()['logo_url']
        for name, group in group_list[:5]:
            n_dish = min(5, group.shape[0])

            dish_rec_df = group.sample(n_dish)
            dish_rec_dict = dish_rec_df.to_dict()
            index = list(dish_rec_df.index)
            rest_dishes = []

            for id in index:
                rest_dishes.append({"url": dish_rec_dict["url"][id],
                                    "name": dish_rec_dict["dish_name"][id],
                                    "score": ""})
            rest_url = restraunt_url.get(name, "")
            reco_dict[name] = {
                "dishes": rest_dishes, "score": "", "rest_url": rest_url}
        return reco_dict

    def recomendation_impl(self, meta_list):
        image_list = []
        for meta in meta_list:
            img_pic = Image.open(BytesIO(meta.im_bytes))
            image_list.append(img_pic)
        image_embeddings = CONTEXT.food_clf.predict(image_list)
        
        # Убираем плохие блюда
        filtered_df = RECO_DF[~RECO_DF['category_name'].str.lower().str.contains("соус")]
        filtered_df = filtered_df[~filtered_df['category_name'].str.lower().str.contains("напит")]
        filtered_df = filtered_df[~filtered_df['category_name'].str.lower().str.contains("детск")]
        filtered_df = filtered_df[~filtered_df['dish_name'].str.lower().str.contains("напит")]
        filtered_df = filtered_df[filtered_df['dish_name'] != 'Ям вун сен му']
        filtered_df = filtered_df[~filtered_df['dish_name'].str.contains("Соус")]
        filtered_df = filtered_df[~filtered_df['dish_name'].str.contains("СОУС")]
        
        reco_rests = get_recommend(filtered_df, image_embeddings)
        reco_dict = {}
        restraunt_url = REST_URL_DF.to_dict()['logo_url']
        for res_data in reco_rests:
            rest_dishes = []
            # TODO: сделать по нормальному
            without_repeat = remove_repeat(res_data.rec_dishes)
            best_dishes = sorted(without_repeat, key=lambda x: x.score)
            
            for dish_meta in best_dishes[:5]:
                rest_dishes.append(
                    {"url": dish_meta.dish_url, "name": dish_meta.dish_name, "score": dish_meta.score})
            rest_url = restraunt_url.get(res_data.rest_name, "")
            reco_dict[res_data.rest_name] = {
                "dishes": rest_dishes, "score": res_data.score, "rest_url": rest_url}
        return reco_dict
