from types import SimpleNamespace
from time import time
import torch as ch
from tqdm import tqdm

class StandardLoop():

    def __init__(self, accelerator, model, criterion,
                 metrics=None, optimizer=None):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics

    def train(self, loader):
        return self.run(loader=loader, mode='train')

    def eval(self, loader):
        return self.run(loader=loader, mode='eval')

    def alter_inputs(self, inputs, labels):
        return inputs, labels

    def pre_loop(self, loader, mode):
        if mode == 'train':
            self.model.train()
            ch.autograd.set_grad_enabled(True)
        elif mode == 'eval':
            self.model.eval()
            ch.autograd.set_grad_enabled(False)
        else:
            raise ValueError()


    def run(self, loader, mode):
        self.pre_loop(loader, mode)

        for original_inputs, original_labels in tqdm(x for x in loader):

            inputs, labels = self.alter_inputs(original_inputs,
                                               original_labels)

            prediction = self.model(inputs)
            loss = self.criterion(prediction, labels)

            if mode == 'train':
                self.accelerator.backward(loss)
                self.optimizer.step()

            self.metrics.log('loss', loss)
            self.metrics.compute(SimpleNamespace(**locals()))

