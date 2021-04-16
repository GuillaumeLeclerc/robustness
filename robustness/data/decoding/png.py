import numpy as np
import warnings
import io
from PIL import Image
import cv2

from .jpeg import decode_PIL, decode_cv2

try:
    import pyspng
except:
    pass

def decode_pyspng(buff):
    if pyspng is None:
        warnings.warn("pyspng unavailable, using cv2", RuntimeWarning)
        return decode_cv2(buff)

    data = pyspng.load(buff)
    return data

default = decode_pyspng
