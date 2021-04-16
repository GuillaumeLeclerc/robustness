from abc import ABCMeta, abstractmethod
from functools import partial

import torch as ch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import webdataset as wds

from .dataset_factory import DatasetFactory
from ..augmentation import Augmenter


def apply_augmenter(sample, factory, transform, transform_label):
    data, labels = factory.unpack_sample(sample)
    return factory.pack_sample(transform(data), transform_label(labels))

def default_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a collection of samples (dictionaries) and create a batch.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], ch.Tensor):
            if combine_tensors:
                import torch
                b = torch.stack(list(b))
        elif isinstance(b[0], int):
            b = ch.LongTensor(b)
        elif isinstance(b[0], float):
            b = ch.FloatTensor(b)
        elif isinstance(b[0], np.ndarray):
            if combine_tensors:
                b = ch.from_numpy(np.stack(b))
        else:
            b = list(b)
        result.append(b)
    return result

class FromWebDatasetFactory(DatasetFactory, metaclass=ABCMeta):

    def __init__(self, shards):
        self.shards = shards

    def generate_sample_dataset(self, augmenter: Augmenter = None,
                 shuffle_size: int = 0) -> IterableDataset:
        dataset = wds.WebDataset(self.shards, shardshuffle=shuffle_size > 0)
        dataset = self.decode(dataset)
        if augmenter:
            dataset = dataset.map(partial(apply_augmenter, factory=self,
                                          transform=augmenter.transform_sample_data,
                                          transform_label=augmenter.transform_sample_label))
        return dataset

    def generate_batched_dataset(self, batch_size, num_workers=0,
                                 augmenter: Augmenter = None,
                                 shuffle_size: int = 0, pin_memory=True):
        dataset = self.generate_sample_dataset(augmenter, shuffle_size)
        dataset = dataset.batched(batch_size)
        dataset = DataLoader(dataset, batch_size=None, num_workers=num_workers,
                             pin_memory=pin_memory)
        if augmenter:
            dataset = (wds.Processor(dataset, lambda x: x)
                       .map(partial(apply_augmenter, factory=self,
                                     transform=augmenter.batch_transform_data,
                                     transform_label=augmenter.batch_transform_label))
                       )
        return dataset

    @abstractmethod
    def decode(self, dataset):
        raise NotImplementedError

