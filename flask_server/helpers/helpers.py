from dataclasses import dataclass
from io import BytesIO
from redis_collections import Dict
from redis import StrictRedis
from PIL import Image
import redis
import os


@dataclass
class ImageMeta:
    im_bytes: bytes
    im_b64: str
    caption: str


def resize_im_bytes(image_bytes, size=256):
    im = Image.open(BytesIO(image_bytes))
    im = im.resize((size, size))
    img_byte_arr = BytesIO()
    im.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def im_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


class PostStorage(object):
    def __init__(self, key='4ee69ce4970b4580bc80acac2572f16e') -> None:
        super().__init__()
        host=os.environ.get("REDIS")
        if host is None:
            host="localhost"
        redis = StrictRedis(host=host)
        self.images_meta = Dict(
            key='4ee69ce4970b4580bc80acac2572f16e', redis=redis)
        if redis.exists("id"):
            self.id = int(redis.get("id"))
        else:
            self.id = 0
        self.key = key
        self.redis = redis

    def add(self, val: ImageMeta):
        self.images_meta[self.id] = val
        self.id += 1
        self.redis.set("id", self.id)

    def remove(self, id):
        self.images_meta.pop(id)

    def clean(self):
        self.images_meta.clear()
        self.id = 0
        self.redis.set("id", 0)
