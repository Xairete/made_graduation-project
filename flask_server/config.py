import os


class Config(object):
    DETECTOR_MODEL_NAME = os.environ["DETECTOR_MODEL_NAME"]
    CLASSIFIER_FNAME = os.environ["CLASSIFICATOR_MODEL_NAME"]
    SELECTOR_FNAME = os.environ["SELECTOR_MODEL_NAME"]
