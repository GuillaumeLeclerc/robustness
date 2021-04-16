from glob import glob
from os import path

import torch as ch
import webdataset as wds
import kornia.augmentation as aug
from tqdm import tqdm

from .from_webdataset_generator import FromWebDatasetFactory
from ..augmentation import Augmenter
from ..decoding.jpeg import default as jpeg_decode
from ..decoding.png import default as png_decode


# from fastargs import Section, Param

class ImageNetWDS(FromWebDatasetFactory):

    def __init__(self, folder):
        shards = glob(path.join(folder, '*.tar'))
        super().__init__(shards)

    def decode(self, dataset):
        return (
            dataset
            .decode(wds.handle_extension('.jpg', jpeg_decode))
            .decode(wds.handle_extension('.jpeg', jpeg_decode))
            .decode(wds.handle_extension('.png', png_decode))
            .rename(image="jpg;jpeg;png", cls="cls")
            .map_dict(image=lambda x: (x.astype('float32') * 255).transpose(2, 0, 1))
            .to_tuple("image", "cls")
            .map_tuple(ch.from_numpy, lambda x: x)

        )

class ClassicINTrainingTransforms(Augmenter):

    def transform_sample_data(self, sample):
        ch.set_num_threads(1)
        return aug.RandomResizedCrop((224, 224))(sample).squeeze()
