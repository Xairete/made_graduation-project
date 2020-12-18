import os
DETECTOR_MODEL_NAME="models/food_non_food_classifier_models_resnet18.pth"
CLASSIFICATOR_MODEL_NAME="models/resnet50-new_weights.pth"
SELECTOR_MODEL_NAME="models/detection_model.pt"


class Config(object):
    DETECTOR_MODEL_NAME = DETECTOR_MODEL_NAME
    CLASSIFIER_FNAME = CLASSIFICATOR_MODEL_NAME
    SELECTOR_FNAME = SELECTOR_MODEL_NAME