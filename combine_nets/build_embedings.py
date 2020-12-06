# %%
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import json


# %%

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_embed_model(use_gpu=False):
    NUM_OF_CLASSES = 308

    model = models.resnet50(pretrained=False)

    # I recommend training with these layers unfrozen for a couple of epochs after the initial frozen training
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)

    if use_gpu:
        model = model.cuda()
        model.load_state_dict(torch.load("resnet50-new_weights.pth"))
    else:
        model.load_state_dict(torch.load("resnet50-new_weights.pth", map_location=torch.device('cpu')))

    class_predictor = model.fc
    model.fc = Identity()

    return model, class_predictor


class ImagePredictor:
    def __init__(self):
        model, class_predictor = get_embed_model(use_gpu=False)
        self.model = model
        self.class_predictor = class_predictor
        self.test_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.num_to_class = json.load(open('num_to_class.json', 'r'))

    def predict_image(self, image):
        self.model.train(False)
        self.class_predictor.train(False)

        input_ = image.convert('RGB')
        input_ = self.test_transforms(input_)
        
        embeds = self.model(input_[None, ...])
        _, pred = torch.max(self.class_predictor(embeds).data, 1)

        class_name = self.num_to_class[str(pred.item())]

        return embeds, class_name

# %%
