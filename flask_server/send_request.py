import requests


if __name__ == "__main__":
    url = 'http://127.0.0.1:5000/predict'
    multiple_files = [
    ('images', ('image_104.jpg', open('data/image_104.jpg', 'rb'))),
    ('images', ('image_106.jpg', open('data/image_106.jpg', 'rb')))]
    r = requests.post(url, files=multiple_files)
    r.text
