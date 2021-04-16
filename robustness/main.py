import torch as ch

from .internal  import CudaScheduler, StandardLoop
# from .data.datasets import ImageNetWDS
from .data.datasets.imagenet_wds import ImageNetWDS, ClassicINTrainingTransforms


folder = "/data2/datasets/imagenet-webdataset/"

ds = ImageNetWDS(folder).generate_batched_dataset(500, augmenter = ClassicINTrainingTransforms(), num_workers=6)

model = ch.nn.Linear(10, 10)
optimizer = ch.optim.SGD(model.parameters(), lr=0.1)

criterion = ch.nn.CrossEntropyLoss()

scheduler = CudaScheduler({
    'train': StandardLoop(criterion, optimizer),
    'test': StandardLoop(criterion),
}, {
    'train': ds,
    'test': ds
})

scheduler.run(100)

