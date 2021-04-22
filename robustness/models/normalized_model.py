import torch as ch
from torchvision.transforms import Normalize

class NormalizedModel(ch.nn.Sequential):

    def __init__(self, model, std, var):
        normalizer = Normalize(std, var)
        super().__init__(normalizer, model)
