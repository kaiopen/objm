from typing import Any, Dict, Optional, Tuple

from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat, TorchReal
from kaitorch.zoo.droppath import DropPath
from kaitorch.zoo.mamba import Mamba, MambaBlock
from kaitorch.zoo.simba import EinFFT


ATTN = {
    'Mamba': Mamba,
    'MambaBlock': MambaBlock
}
FFN = {
    'EinFFT': EinFFT
}


class Former(nn.Module):
    r'''

    #### Args:
    - in_channels: the number of input channels.
    - attn: the type of an attention layer.
    - attn_kw: the arguments to the attention layer.
    - ffn: the type of a FFN layer.
    - ffn_kw: the arguments to the FFN layer.
    - drop: the probability of doing dropping a path.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a tensor. Its shape should be `(B, L, C)`.

    #### Returns:
    - A tensor. Its shape is `(B, L, C)`.

    '''
    def __init__(
        self,
        in_channels: int,
        attn: str = 'Mamba',
        attn_kw: Optional[Dict[str, Any]] = None,
        ffn: str = 'EinFFT',
        ffn_kw: Optional[Dict[str, Any]] = None,
        drop: float = 0.
    ) -> None:
        super().__init__()
        self._attn = ATTN[attn](in_channels=in_channels, **attn_kw)
        self._ffn = FFN[ffn](in_channels=in_channels, **ffn_kw)
        self._norm_attn = nn.LayerNorm(in_channels)
        self._norm_ffn = nn.LayerNorm(in_channels)
        self._drop = DropPath(drop)

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x = x + self._drop(self._attn(self._norm_attn(x)))
        return x + self._drop(self._ffn(self._norm_ffn(x)))


class Former2d(Former):
    def __init__(
        self,
        in_channels: int,
        attn: str = 'Mamba',
        attn_kw: Optional[Dict[str, Any]] = None,
        ffn: str = 'EinFFT',
        ffn_kw: Optional[Dict[str, Any]] = None,
        scan: str = 'H+',
        drop: float = 0.
    ) -> None:
        super().__init__(in_channels, attn, attn_kw, ffn, ffn_kw, drop)
        match scan:
            case 'H+':
                self._column_first = False
                self._reversed = False
            case 'H-':
                self._column_first = False
                self._reversed = True
            case 'W+':
                self._column_first = True
                self._reversed = False
            case 'W-':
                self._column_first = True
                self._reversed = True
            case _:
                raise ValueError(f'An invalid scanning method ({scan}).')

    def flatten(
        self, x: TorchTensor[TorchReal]
    ) -> Tuple[TorchTensor[TorchReal], Tuple[int, int, int, int]]:
        r'''

        #### Args:
        - x: A feature map. Its shape should be `(B, C, H, W)`.

        #### Returns:
        - A flattened feature map. Its shape is `(B, H * W, C)`.
        - A shape in the form of `(B, C, H, W)` or `(B, C, W, H)`.

        '''
        if self._column_first:
            x = x.transpose(2, 3)
        shape = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        if self._reversed:
            x = x.flip(1)
        return x, shape

    def reshape(
        self,
        x: TorchTensor[TorchReal],
        shape: Tuple[int, int, int, int]
    ) -> TorchTensor[TorchReal]:
        r'''

        #### Args:
        - x: a feature map. Its shape should be `(B, H * W, C)`.
        - shape: a target shape. It should be in the form of `(B, C, H, W)` or
            `(B, C, W, H)`.

        #### Returns:
        - A reshaped feature map. Its shape is `(B, C, H, W)`.

        '''
        if self._reversed:
            x = x.flip(1)
        x = x.transpose(1, 2).reshape(shape)
        if self._column_first:
            x = x.transpose(2, 3)
        return x

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        x, shape = self.flatten(x)
        return self.reshape(super().forward(x), shape)
