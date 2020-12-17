import sys

import pandas as pd
from config import Config
from ml_engine.food_classifier import FoodClassifier
from ml_engine.food_detector import FoodDetector
from ml_engine.food_selector import FoodSelector
from sqlalchemy.engine import create_engine

from helpers.helpers import PostStorage

sys.path.append("..")


class AppContext(object):
    food_clf = FoodClassifier(Config.CLASSIFIER_FNAME)
    food_detector = FoodDetector(Config.DETECTOR_MODEL_NAME)
    food_selector = FoodSelector(Config.SELECTOR_FNAME)


CONTEXT = AppContext()
POST_STORAGE = PostStorage()
ENGINE = create_engine('sqlite:///df/meta.db')
RECO_DF = pd.read_sql_table(
    'reco',
    con=ENGINE,
    index_col=['index'],
)
REST_URL_DF = pd.read_sql_table(
    'rest_url',
    con=ENGINE,
    index_col=['restaurant_name']
)
