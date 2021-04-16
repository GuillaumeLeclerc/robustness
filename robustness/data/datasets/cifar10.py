from glob import glob
from os import path

import webdataset as wds

from .from_webdataset_generator import FromWebDatasetGenerator
from ..decoding.jpeg import default as jpeg_decode
from ..decoding.png import default as png_decode


# from fastargs import Section, Param

folder = "/data2/datasets/imagenet-webdataset/"

class ImageNetWDS(FromWebDatasetGenerator):

    def __init__(self, folder):
        shards = glob(path.join(folder, '*.tar'))
        super().__init__(shards)

    def decode(self, dataset):
        return (
            dataset
           .decode(wds.handle_extension('.jpg', jpeg_decode))
           .decode(wds.handle_extension('.jpeg', jpeg_decode))
           .decode(wds.handle_extension('.png', png_decode))
           .to_tuple("jpg;jpeg;png", "cls")
        )

ds = ImageNetWDS(folder).generate()
print(next(iter(ds)))
