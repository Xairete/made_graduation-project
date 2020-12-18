import base64
import heapq
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class DishMeta:
    score: np.float
    insta_index: int
    dish_index: np.int64
    dish_url: str = ""
    dish_name: str = ""


@dataclass
class RestMeta:
    rec_dishes: List[DishMeta]
    score: np.float
    rest_name: str


def calc_restaraunt_score(rest_emb, insta_embed, mean_window=5, num_dish=2) -> Tuple[np.float, List[DishMeta]]:
    if len(rest_emb) == 0:
        return float('inf'), []

    matches = []

    rest_embed_values=np.array([np.frombuffer(row, dtype=np.float32) for row in rest_emb.values])
    rest_embed_index = rest_emb.index
    for i, insta_data in enumerate(insta_embed):
        insta_emb = insta_data

        dist = rest_embed_values-insta_emb.numpy().reshape(-1)
        dist = np.linalg.norm(dist, axis=1)
        if dist.shape[0] > num_dish:
            min_id = np.argpartition(dist, num_dish)[:num_dish]
        else:
            min_id = list(range(dist.shape[0]))

        for id in min_id:
            matches.append(DishMeta(
                score=dist[id],
                insta_index=i,
                dish_index=rest_embed_index[id]
            ))

    n_smallest = heapq.nsmallest(mean_window, matches, key=lambda x: x.score)
    cumm_emb = [match_data.score for match_data in n_smallest]

    return np.mean(cumm_emb), matches


def get_recommend(df, insta_embeddings) -> List[RestMeta]:
    rest_groups = df.groupby('restaurant_name')
    final_result = []
    for name, rest_group in rest_groups:
        # Убираем плохие рестораны
        if name in ['Хинкали vs Хачапури', 'Шерлок', 'Виноградник', 'Дорогомилово Сервис']:
            continue
        
        emb = rest_group['embed']
        mean_dist, matches = calc_restaraunt_score(emb, insta_embeddings)

        loc_index = [match.dish_index for match in matches]
        dishes_data = rest_group[['url','dish_name']].loc[loc_index].to_dict()

        for match in matches:
            key = match.dish_index
            match.dish_url = dishes_data['url'][key]
            match.dish_name = dishes_data['dish_name'][key]

        final_result.append(
            RestMeta(score=mean_dist, rest_name=name, rec_dishes=matches))
    n_smallest = heapq.nsmallest(5, final_result,  key=lambda x: x.score)
    return n_smallest


"""
TODO: собрать в одно место
"""


def im_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def get_b64_images(result):
    images_list = []
    for res in result[1]:
        im = Image.open(res['name'])
        bytes_im = im_to_bytes(im)
        b64_im = base64.b64encode(
            bytes_im).decode('ascii')
        images_list.append(b64_im)
    return images_list
