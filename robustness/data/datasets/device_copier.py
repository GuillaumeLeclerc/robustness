from ..augmentation.augmenter import Augmenter, apply_augmenter

def  deviceCopierIterator(dataset, device, factory, augmenter: Augmenter):
    for batch in dataset:
        batch = tuple(x.to(device, non_blocking=True) for x in batch)
        if augmenter is not None:
            batch = apply_augmenter(batch, factory=factory,
                                    transform=augmenter.transform_batch_data,
                                    transform_label=augmenter.transform_batch_label)

        yield batch


class DeviceCopier:

    def new_iter(self):
        return iter(deviceCopierIterator(self.dataset, self.device,
                                         self.factory,
                                         self.augmenter))

    def __init__(self, dataset, device, factory, augmenter:Augmenter = None,
                 one_ahead=False):
        self.dataset = dataset
        self.device = device 
        self.factory = factory
        self.augmenter = augmenter
        if one_ahead:
            self.next_iter = self.new_iter()
        else:
            self.next_iter = None

    def __iter__(self):
        new_iter = self.new_iter()

        if self.next_iter is not None:
            new_iter, self.next_iter = self.next_iter, new_iter

        return new_iter
