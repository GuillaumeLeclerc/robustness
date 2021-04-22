import torch as ch

def compute_generic_normalization(generator):
    # TODO make it numerically stable
    s1 = None
    s2 = None
    pixels = 0
    for data in generator:
        p_sum = data.sum(dim=(0, 2))
        p_sum2 = (data ** 2).sum(dim=(0, 2))
        pixels += data.shape[0] * data.shape[2]
        if s1 is None:
            s1 = p_sum
            s2 = p_sum2
        else:
            s1 += p_sum
            s2 += p_sum2
    mean = s1 / pixels
    std = ch.sqrt((s2 / pixels) - mean**2)

    return mean, std


def image_classification_normalization(dataset):

    def extractor():
        for img, label in dataset:
            yield img.view(img.shape[0], img.shape[1], -1)

    return compute_generic_normalization(extractor())

