from typing import Any, Dict, Sequence, Union
import math

import torch
from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat, TorchInt64, TorchReal

from .former import Former


def decode(
    proposal: TorchTensor[TorchReal], k: float = 0.8
) -> TorchTensor[TorchInt64]:
    proposal = proposal.flatten(1)  # (B, HW)
    return torch.topk(
        proposal,
        k=max(math.ceil(proposal.shape[1] * k), 1),
        dim=-1,
        largest=True,
        sorted=True
    )[1]  # (B, k)


class MFAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        former: Dict[str, Any],
        depth: int,
        k: float = 0.8,
        drops: Union[float, Sequence[float]] = 0.,
        *args, **kwargs
    ):
        r'''Mamba-based foreground attention mechanism

        '''
        super().__init__()
        if depth <= 0:
            raise ValueError('`depth` should be larger than 0.')

        if isinstance(drops, float):
            drops = [drops] * depth
        elif (d := len(drops)) != depth:
            raise ValueError(
                f'The length of `drops` ({d}) should be equal to'
                f' `depth` ({depth}).'
            )

        self._formers = nn.Sequential(
            *[Former(in_channels=in_channels, drop=p, **former) for p in drops]
        )

        self._k = k

    def forward(
        self, x: TorchTensor[TorchFloat], proposal: TorchTensor[TorchReal]
    ) -> TorchTensor[TorchFloat]:
        shape = x.shape
        indices = decode(proposal, self._k)
        b = torch.arange(
            shape[0], device=x.device
        ).unsqueeze_(-1).expand_as(indices)

        x = x.flatten(2).transpose(1, 2)
        return torch.index_put(
            x, (b, indices), self._formers(x[b, indices])
        ).transpose(1, 2).reshape(shape)
