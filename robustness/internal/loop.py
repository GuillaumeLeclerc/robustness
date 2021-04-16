from abc import ABC, ABCMeta, abstractmethod


class Loop(ABC):

    @abstractmethod
    def pre_compute(self, it, data, labels):
        raise NotImplementedError

    @abstractmethod
    def compute(self, it, model, data, labels):
        raise NotImplementedError

    @abstractmethod
    def post_compute(self, it, outputs):
        raise NotImplementedError


def copy_to_cuda(data):
    if isinstance(data, tuple):
        return tuple(x.cuda(non_blocking=True) for x in data)

    return copy_to_cuda((data,))[0]

class StandardLoop(Loop, metaclass=ABCMeta):

    def __init__(self, criterion, optimizer=None, lr_scheduler=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def pre_compute(self, it, data, labels):
        data = copy_to_cuda(data)
        labels = copy_to_cuda(labels)

        return data, labels

    def compute(self, it, model, data, labels):
        # Set the mode of the model
        if it == 0:
            if self.optimizer is None:
                model.eval()
            else:
                model.train()

        # Forward
        outputs = model(data)
        loss = self.criterion(outputs, labels)

        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backwards()
            self.optimizer.step()

        return outputs


    def post_compute(self, it, outputs):
        pass








