import pandas as pd
import numpy as np
from io import BytesIO
import base64
from PIL import Image

def calc_restaraunt_score(rest_emb,insta_embed, mean_window=5):
    if len(rest_emb) == 0:
        return float('inf'), []
    
    matches = []
    
    for i, insta_data in enumerate(insta_embed):
        insta_emb = insta_data
        
        dist = rest_emb-insta_emb.numpy().reshape(-1)
        dist = np.linalg.norm(dist, axis=1)
        min_id = np.argmin(dist)
        
        matches.append({
            'score': dist[min_id],
            'id': min_id,
            'insta_index': i
        })
    
    matches = list(sorted(matches, key=lambda x: x['score']))
    matches = matches[:mean_window]
    
    sum_ = 0
    
    for match_data in matches:
        sum_ += match_data['score']
        
    return sum_ / mean_window, matches

def get_recommend(df, insta_embeddings):
    rest_groups=df.groupby('rest_name')
    final_result = []
    for name, rest_emb in rest_groups:
        emb = rest_emb.iloc[:,:2048].values
        result = calc_restaraunt_score(emb, insta_embeddings)
        for res in result[1]:
            res['name']=rest_emb['image_path'].iloc[res['id']]
        final_result.append(result)

    return min(final_result)

"""
TODO: собрать в одно место
"""
def im_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def get_b64_images(result):
    images_list =[]
    for res in result[1]:
        im = Image.open(res['name'])
        bytes_im = im_to_bytes(im)
        b64_im = base64.b64encode(
                        bytes_im).decode('ascii')
        images_list.append(b64_im)
    return images_list
