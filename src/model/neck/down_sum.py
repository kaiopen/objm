from typing import Any, Dict, List, Sequence, Tuple, Union

from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.nn.conv import Conv2dBlock
from kaitorch.zoo.resnet import Bottleneck


class DownSum(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        if (n := len(in_channels)) != (m := len(out_channels)):
            raise ValueError(
                f'`in_channels` ({n}) and `out_channels` ({m}) should share'
                ' the same length.'
            )

        self._blocks = nn.ModuleList()
        for in_c, out_c in zip(in_channels, out_channels):
            self._blocks.append(
                Bottleneck(
                    in_channels=in_c,
                    out_channels=out_c,
                    padding_mode=padding_mode,
                    activation=activation,
                    activation_kw=activation_kw
                )
            )

        self._downs = nn.ModuleList()
        for i, c in enumerate(out_channels[:-1]):
            self._downs.append(
                nn.Sequential(
                    Conv2dBlock(
                        c, c, 3, 2, 1,
                        groups=c,
                        padding_mode=padding_mode,
                        mode='cn'
                    ),
                    Conv2dBlock(
                        c, out_channels[i + 1], 1, 1, 0,
                        activation=activation,
                        activation_kw=activation_kw
                    )
                )
            )

        self.apply(self.init_weights)

    def forward(
        self,
        x: Tuple[List[TorchTensor[TorchFloat]], Any]
    ) -> Tuple[List[TorchTensor[TorchFloat]], Any]:
        xs, aux = x
        out = self._blocks[0](xs[0])
        xs[0] = out

        for i, (x, block, down) in enumerate(
            zip(xs[1:], self._blocks[1:], self._downs), start=1
        ):
            out = block(x) + down(out)
            xs[i] = out
        return xs, aux

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
