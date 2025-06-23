from torch.nn import Identity

from .down_sum import DownSum


NECK = {
    'Identity': Identity,
    'DownSum': DownSum
}
