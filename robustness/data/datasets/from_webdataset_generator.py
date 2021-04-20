from abc import ABCMeta, abstractmethod
from functools import partial

import torch as ch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import webdataset as wds
import random

from .dataset_factory import DatasetFactory
from .device_copier import DeviceCopier
from ..augmentation.augmenter import Augmenter, apply_augmenter

def worker_init(self):
    ch.set_num_threads(1)

def custom_collation_fn(samples, combine_tensors=True, combine_scalars=True):
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


class EpochSeparator:

    def __init__(self, dataset):
        self.it = iter(dataset)
        self.epoch = -1
        self.value = None

    def __iter__(self):
        self.epoch += 1
        return self

    def __next__(self):
        result = None
        if self.value is not None:
            result = self.value
            self.value = None
        else:
            result = next(self.it)

        epoch, data = result
        if epoch > self.epoch:
            self.value = result
            raise StopIteration()
        else:
            return data

class EpochAdder(IterableDataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
        self.it = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            n = next(self.it)
            return self.epoch, n
        except StopIteration:
            self.epoch += 1
            self.it = iter(self.dataset)
            return next(self)


class FromWebDatasetFactory(DatasetFactory, metaclass=ABCMeta):

    def __init__(self, shards):
        self.shards = shards

    def generate_sample_dataset(self, augmenter: Augmenter = None,
                 shuffle_size: int = 0) -> IterableDataset:
        dataset = wds.WebDataset(self.shards, shardshuffle=shuffle_size > 0)

        dataset = dataset.then(wds.iterators.shuffle, shuffle_size,
                               shuffle_size)

        dataset = self.decode(dataset)

        return dataset


    def generate_batched_dataset(self, batch_size, num_workers=0,
                                 augmenter: Augmenter = None,
                                 shuffle_size: int = 0, pin_memory=True,
                                 device='cpu'):
        num_workers = min(num_workers, len(self.shards))
        dataset = self.generate_sample_dataset(augmenter, shuffle_size)
        dataset = dataset.then(wds.iterators.batched, batchsize=batch_size, collation_fn=custom_collation_fn)
        dataset = EpochAdder(dataset)

        dataset = DataLoader(dataset, batch_size=None, num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=worker_init,
                             shuffle=False, sampler=None)

        dataset = EpochSeparator(dataset)
        dataset = DeviceCopier(dataset, device, self, augmenter)
        return dataset

    @abstractmethod
    def normalizer(self):
        raise NotImplementedError

    @abstractmethod
    def decode(self, dataset):
        raise NotImplementedError

