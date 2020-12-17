from helpers.states import CONTEXT
from helpers.helpers import ImageMeta
import sys
import string
from typing import List

sys.path.append("..")

MAX_LEN=300

def prep_text(text: str):
    trans = str.maketrans("","",string.punctuation)
    clean_text = text.lower().replace("\n"," ")
    clean_text = " ".join(filter(lambda x: not x.startswith("#"), clean_text.split(" ")))
    clean_text = clean_text.translate(trans)
    return clean_text

def filter_positive_comments(meta: List[ImageMeta]):
    comments = [prep_text(m.caption) if len(m.caption) else "" for m in meta]

    predict = CONTEXT.fasttext_model.predict(comments)
    result = []
    for i in range(len(meta)):
        if predict[0][i][0]=='__label__1':
            result.append(meta[i])
        elif predict[1][i][0] < 0.8:
            result.append(meta[i])
    return result





