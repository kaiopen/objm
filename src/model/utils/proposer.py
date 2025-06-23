from typing import List, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import Conv2dBlock


class MultiProjector(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Optional[int] = None,
        *args, **kwargs
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels[1]
        num_group = min(in_channels)
        self._projs = nn.ModuleList(
            [
                Conv2dBlock(
                    c, out_channels, 1, 1, 0,
                    normalization='groupnorm',
                    normalization_kw={
                        'num_groups': num_group,
                        'num_channels': out_channels
                    },
                    mode='cn'
                ) for c in in_channels
            ]
        )

    def forward(
        self, xs: Sequence[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        return [proj(x) for x, proj in zip(xs, self._projs)]


class MultiScaleProposer(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        *args, **kwargs
    ) -> None:
        super().__init__()
        o = in_channels[1]
        self._projector = MultiProjector(in_channels, o)
        self._mlp = nn.Sequential(
            Conv2dBlock(
                o, o, 1,
                normalization='groupnorm',
                normalization_kw={'num_groups': 1, 'num_channels': o},
                activation='gelu',
                activation_kw={},
                mode='nca'
            ),
            Conv2dBlock(
                o, o, 3, 1, 1,
                groups=o,
                bias=True,
                padding_mode=padding_mode,
                activation='gelu',
                activation_kw={},
                mode='ca'
            ),
            nn.Conv2d(o, 1, 1)
        )  # shared MLP

        num = len(in_channels)
        self._alphas = nn.Parameter(torch.zeros(num - 1)) if num > 1 else None

    def forward(
        self, xs: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        ps = self._projector(xs)

        mlp = self._mlp
        alphas = self._alphas
        ps.reverse()
        g = mlp(ps[0])
        ps[0] = g
        for i, p in enumerate(ps[1:], start=1):
            ps[i] = mlp(
                p + F.interpolate(
                    g, size=p.shape[2:], mode='bilinear', align_corners=True
                ) * alphas[i - 1]
            )
        ps.reverse()
        return ps
