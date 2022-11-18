from docarray import Document, DocumentArray
import imagehash
from PIL import Image
import numpy as np
import os


def get_label_embedding(label_da: DocumentArray, client):
    return client.encode(label_da)


def classify(img: bytes, label_da: DocumentArray, topn: int, client):
    d = Document(blob=img, matches=label_da)
    r = client.rank([d], show_progress=True)
    result = r['@m', ['text', 'scores__clip_score__value']]
    return [each[:topn] for each in result]
