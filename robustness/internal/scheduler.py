from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict


class Scheduler(ABC):

    def __init__(self, loops: OrderedDict, dataset):
        self.loops = loops

    @abstractmethod
    def run(self, max_epochs: int, early_stopping_cond=None):
        pass


class CudaScheduler(Scheduler):

    def run(self, max_epochs: int, early_stopping_cond=None):
        print("I'm running")
