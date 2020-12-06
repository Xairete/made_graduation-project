from instagram import Account, Media, WebAgent #взять отсюда https://github.com/OlegYurchik/pyInstagram
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from typing import List


def download_images(account_name="mama_na_kuxne", max_count=20):

    agent = WebAgent()
    account = Account(account_name)

    link_list = []
    pointer = None

    media, pointer = agent.get_media(account, count=max_count)
    for med in media:
        try:
            link_list.append(med.resources[0])
        except Exception:
            continue

    img_dimensions = 224

    img_transforms = transforms.Compose([
        transforms.Resize((img_dimensions, img_dimensions)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet_loaded = torch.hub.load('pytorch/vision', 'resnet18')
    resnet_loaded.fc = nn.Sequential(nn.Linear(resnet_loaded.fc.in_features, 512), nn.ReLU(), nn.Dropout(),
                                     nn.Linear(512, 2))
    resnet_loaded.load_state_dict(torch.load('../food_non_food_classifier/models/resnet18.pth', map_location=torch.device('cpu')))
    resnet_loaded.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    import requests
    from PIL import Image
    from io import BytesIO

    img_list = []  # здесь будут храниться PIL-image фотографий еды
    food_link_list = []
    for link in link_list:
        try:
            response = requests.get(link, stream=True)
            img_pic = Image.open(BytesIO(response.content))
            img = img_transforms(img_pic)
            img = img.unsqueeze(0)
            resnet_loaded.to(device)
            prediction = resnet_loaded(img.to(device))
            prediction = prediction.argmax()
            if (prediction == 1):
                img_list.append(img_pic)
                food_link_list.append(link)
                print(link)
            del response
        except Exception:
            continue

    return img_list