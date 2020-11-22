#!/usr/bin/env python3

import requests
from furl import furl
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def download(data: dict, root_dir: Path):
    for restaurant_name in tqdm(data):
        restaurant = data[restaurant_name]
        for category_name in restaurant['food_category']:
            category = restaurant['food_category'][category_name]

            restaurant_name = restaurant_name.replace('/', 'or')
            category_name = category_name.replace('/', 'or')

            category_path = root_dir.joinpath(restaurant_name, category_name)
            category_path.mkdir(parents=True, exist_ok=True)

            for dish in category:
                url = dish['img_url']
                if url is not None:
                    url = furl(url).remove(args=True, fragment=True).url
                    dish_name = dish['name']
                    dish_name = dish_name.replace('/', 'or')
                    image_path = category_path.joinpath(dish_name + '.jpg')
                    r = requests.get(url, allow_redirects=True)

                    open(image_path, 'wb').write(r.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download delivery dataset.')

    parser.add_argument('--root', default=Path.cwd().as_posix(), help='Path to download dataset', type=str)
    parser.add_argument('--data', default='small_data_file.json', help='JSON file with metadata about restaurants',
                        type=str)

    args = parser.parse_args()

    root_dir = Path(args.root).joinpath('restaurant_dataset')

    data = json.load(open(args.data, 'rb'))
    download(data, root_dir)
