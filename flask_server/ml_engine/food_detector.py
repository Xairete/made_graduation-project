import sys
from typing import List

import torch
from helpers.helpers import ImageMeta
from torchvision import models, transforms

from .base_model import BaseModel

sys.path.append("..")

N_CLASS = 2
IMG_DIMENTIONS = 224
FOOD_LABEL = 1


class FoodDetector(BaseModel):
    """
    Определяет тип еды на фото
    TODO: сетка для извлечения эмбедов
    """

    def _init_model(self, model_fname) -> torch.nn.Module:
        model = models.resnet18(pretrained=False)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, N_CLASS)
        )
        model.load_state_dict(torch.load(
            model_fname, map_location=self.device))
        model.eval()
        return model

    def _get_transform_pipeline(self):
        valid_transf = transforms.Compose([
            transforms.Resize((IMG_DIMENTIONS, IMG_DIMENTIONS)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        return valid_transf

    def __init__(self, model_fname):
        super().__init__(model_fname)

    def predict(self, images):
        with torch.no_grad():
            images = [
                self.img_transform(image).unsqueeze_(0) for image in images
            ]
            images = torch.cat(images, 0)
            outputs = self.model(images)
            _, preds = torch.max(outputs.data, 1)
            preds_class = preds.numpy()
        return preds_class


def select_food(meta: List[ImageMeta], labels):
    result = []
    for i in range(len(labels)):
        if labels[i] == FOOD_LABEL:
            result.append(meta[i])
    return result
