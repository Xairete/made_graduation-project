from dataclasses import dataclass

@dataclass
class ImageMeta:
    im_bytes: bytes
    im_b64: str
    caption: str

class FakeDb(object):
    def __init__(self) -> None:
        super().__init__()
        self.images_meta = {}
        self.id = 0

    def add(self, val: ImageMeta):
        self.images_meta[self.id] = val
        self.id += 1

    def remove(self, id):
        self.images_meta.pop(id)
    
    def clean(self):
        self.images_meta = {}