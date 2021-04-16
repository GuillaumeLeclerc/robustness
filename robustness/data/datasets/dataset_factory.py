from abc import ABC, abstractmethod
from typing import Tuple

def unpack_indices(data, indices):
    if len(indices) == 1:
        return data[indices[0]]
    return tuple(data[x] for x in indices)

def pack_indices(result, indices, data):
    if len(indices) == 1:
        result[indices[0]] = data
    else:
        for i, ix in enumerate(indices):
            result[ix] = data[i]


class DatasetFactory(ABC):

    def get_input_indices(self) -> Tuple[int, ...]:
        return (0,)

    def get_label_indices(self) -> Tuple[int, ...]:
        return (1,)

    def unpack_sample(self, sample: Tuple) -> Tuple:
        data = unpack_indices(sample, self.get_input_indices())
        labels = unpack_indices(sample, self.get_label_indices())
        return data, labels

    def pack_sample(self, data, labels) -> Tuple:
        iix = self.get_input_indices()
        lix = self.get_label_indices()
        result = [None] * (len(iix) + len(lix))
        pack_indices(result, iix, data)
        pack_indices(result, lix, labels)
        return tuple(result)
