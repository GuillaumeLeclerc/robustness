from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

def apply_augmenter(sample, factory, transform, transform_label):
    data, labels = factory.unpack_sample(sample)
    r = transform(data)
    r = factory.pack_sample(transform(data), transform_label(labels))
    return r

class Augmenter(ABC):
    """Describe a data augmentation pipeline

    It seperates the operations to be performed on a single sample (CPU)
    from what needs to be done on a complete batch (most of the time on the
    GPU)

    """

    def transform_sample_data(self, *data):
        if len(data) == 1:
            return data[0]
        return data


    def transform_sample_label(self, *label):
        if len(label) == 1:
            return label[0]
        return label

    def transform_batch_data(self, *data_batch):
        if len(data_batch) == 1:
            return data_batch[0]
        return data_batch

    def transform_batch_label(self, *label_batch):
        if len(label_batch) == 1:
            return label_batch[0]
        return label_batch
