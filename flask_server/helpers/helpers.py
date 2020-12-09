class FakeDb(object):
    def __init__(self) -> None:
        super().__init__()
        self.images = {}
        self.id = 0

    def add(self, image, comment):
        self.images[self.id] = ((image, comment))
        self.id += 1

    def remove(self, id):
        self.images.pop(id)