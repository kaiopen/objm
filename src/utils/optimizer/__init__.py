import torch


OPTIMIZER = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'NAdam': torch.optim.NAdam,
    'AdamW': torch.optim.AdamW
}
