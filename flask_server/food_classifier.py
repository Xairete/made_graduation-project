import PIL
import torch
from PIL import Image
import numpy as np
from torchvision import models, transforms

N_CLASS = 308
MODEL_NAME = "resnet50-new_weights.pth"


class FoodClassifier(object):

    def _init_model(self) -> torch.nn.Module:
        model = models.resnet50(pretrained=False)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, N_CLASS)
        model.load_state_dict(torch.load(MODEL_NAME, map_location='cpu'))
        return model

    def _get_transform_pipeline(self):
        valid_transf = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
        return valid_transf

    def __init__(self):
        self.model = self._init_model()
        self.img_transform = self._get_transform_pipeline()
    
    def predict_class(self, image):
        image = self.img_transform(image)
        image.unsqueeze_(0)
        outputs = self.model(image)
        _, preds = torch.max(outputs.data, 1)
        preds_class = preds.numpy()
        return preds_class
