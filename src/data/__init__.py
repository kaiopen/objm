from .imagenet import Dataset, \
    preprocess as preprocess_in, \
    config_ as config_in
from .tiny_imagenet import preprocess as preprocess_tin, \
    config_ as config_tin


CONFIG = {
    'ImageNet': config_in,
    'Tiny-ImageNet': config_tin
}

PREPROCESS = {
    'ImageNet': preprocess_in,
    'Tiny-ImageNet': preprocess_tin
}

DATASET = {
    'ImageNet': Dataset,
    'Tiny-ImageNet': Dataset
}
