from abc import ABCMeta, abstractclassmethod
from torchvision import models, transforms
import torch


class BaseModel(object):
    """
    Базовая модель, грузит сеточку и выдает предикт
    """
    __metaclass__ = ABCMeta

    @abstractclassmethod
    def _init_model(self, model_fname) -> torch.nn.Module:
        pass

    @abstractclassmethod
    def _get_transform_pipeline(self):
        pass

    def __init__(self, model_fname):
        self.model = self._init_model(model_fname)
        self.img_transform = self._get_transform_pipeline()

    @abstractclassmethod
    def predict(self, image):
        pass
