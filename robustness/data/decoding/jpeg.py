import numpy as np
import warnings
import io
from PIL import Image
import cv2

try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
except:
    jpeg = None

def decode_turbojpeg(buff):
    if jpeg is None:
        warnings.warn("TurboJPEG unavailable, using cv2", RuntimeWarning)
        return decode_cv2(buff)

    data = jpeg.decode(buff)
    return data


def decode_cv2(buff):
    buff = np.asarray(bytearray(buff))
    data = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return data

def decode_PIL(buff):
    return np.array(Image.open(io.BytesIO(buff)))

default = decode_turbojpeg
