#!/usr/bin/env python3

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from furl import furl
from tqdm import tqdm

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
NUM_WORKERS = 4


def download(data: dict, root_dir: Path):
    n_rest = len(data)
    ctrl_pnt_path = root_dir.joinpath("ctrl_pnt")
    ctrl_points = bytearray()
    if ctrl_pnt_path.exists():
        ctrl_points = ctrl_pnt_path.read_bytes()
        ctrl_points = bytearray(ctrl_points)
    if not n_rest == len(ctrl_points):
        ctrl_points = bytearray([0]*n_rest)

    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    for idx, restaurant_name in tqdm(enumerate(data), total=n_rest):
        if ctrl_points[idx] == 0:
            restaurant = data[restaurant_name]
            logging.info(
                "start parse restaurant \"{}\"".format(restaurant_name))
            update_restaurant(restaurant, restaurant_name, root_dir, pool)
            ctrl_points[idx] = 1
            ctrl_pnt_path.write_bytes(ctrl_points)
            logging.info("parsed restaurant \"{}\"".format(restaurant_name))
        else:
            logging.debug(
                "restaurant \"{}\" parsed before".format(restaurant_name))
    pool.shutdown()


def update_restaurant(restaurant, restaurant_name, root_dir, pool: ThreadPoolExecutor):
    for category_name in restaurant['food_category']:
        category = restaurant['food_category'][category_name]

        restaurant_name = restaurant_name.replace('/', 'or')
        category_name = category_name.replace('/', 'or')

        category_path = root_dir.joinpath(restaurant_name, category_name)
        category_path.mkdir(parents=True, exist_ok=True)

        def update_url(arg):
            url, image_path = arg
            try:
                r = requests.get(url, allow_redirects=True,
                                 headers=HEADERS, timeout=10)
                return r.content, image_path
            except Exception as ex:
                logging.error(
                    "while parsing url=\"{url}\" path=\"{path}\" got: {ex}".format(url=url, path=image_path, ex=ex))
            return None

        update_list = []
        for dish in category:
            url = dish['img_url']
            if url is not None:
                url = furl(url).remove(args=True, fragment=True).url
                dish_name = dish['name']
                dish_name = dish_name.replace('/', 'or')
                image_path = category_path.joinpath(dish_name + '.jpg')
                update_list.append((url, image_path))
                if len(update_list) == NUM_WORKERS:
                    get_images(pool, update_url, update_list)
        if len(update_list):
            get_images(pool, update_url, update_list)


def get_images(pool, update_url, update_list):
    result = pool.map(update_url, update_list)
    for result in result:
        if result:
            content, image_path = result
            open(image_path, 'wb').write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download delivery dataset.')

    parser.add_argument('--root', default=Path.cwd().as_posix(),
                        help='Path to download dataset', type=str)
    parser.add_argument('--data', default='small_data_file.json', help='JSON file with metadata about restaurants',
                        type=str)
    logging.basicConfig(filename='restaurant_parser.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    args = parser.parse_args()

    root_dir = Path(args.root).joinpath('restaurant_dataset')

    data = json.load(open(args.data, 'rb'))
    download(data, root_dir)
