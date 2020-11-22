import torch
from torchvision import models, transforms
from .base_model import BaseModel

N_CLASS = 308


class FoodClassifier(BaseModel):
    """
    Модель определяет, что на фото еда/не еда
    """
    def _init_model(self, model_fname) -> torch.nn.Module:
        model = models.resnet50(pretrained=False)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, N_CLASS)
        model.load_state_dict(torch.load(model_fname, map_location='cpu'))
        model.eval()
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

    def __init__(self, model_fname):
        super().__init__(model_fname)

    def predict(self, images):
        images = [
            self.img_transform(image).unsqueeze_(0) for image in images
        ]
        images = torch.cat(images, 0)
        outputs = self.model(images)
        _, preds = torch.max(outputs.data, 1)
        preds_class = preds.numpy()
        return preds_class
