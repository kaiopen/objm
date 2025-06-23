from torch.nn import Identity

from .avg_linear import AvgLinear


HEADER = {
    'Identity': Identity,
    'AvgLinear': AvgLinear
}
