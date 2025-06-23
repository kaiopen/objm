from typing import Any, List, Tuple

from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat


class AvgLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, out_channels)
        )

        self.apply(self.init_weights)

    def forward(
        self,
        x: Tuple[List[TorchTensor[TorchFloat]], Any]
    ) -> Tuple[TorchTensor[TorchFloat], Any]:
        xs, aux = x
        return self._layer(xs[-1]), aux

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
