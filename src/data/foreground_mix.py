from typing import Tuple
import random

import torch

from kaitorch.typing import TorchTensor, TorchReal


class ForegroundMix:
    def __init__(self, t: float = 0., p: float = 0.5) -> None:
        self._t = t
        self._p = p

    def __call__(
        self,
        img: TorchTensor[TorchReal],
        label: TorchTensor[TorchReal],
        map: TorchTensor[TorchReal]
    ) -> Tuple[
        TorchTensor[TorchReal], TorchTensor[TorchReal], TorchTensor[TorchReal]
    ]:
        if random.random() < self._p:
            img_0 = img.clone()
            map_0 = map.clone()

            # Remove original foreground.
            mask = map > self._t  # (B, 1, H, W)
            img[mask.expand_as(img)] = 0
            map[mask] = 0

            indices = torch.randperm(img.shape[0])
            mask = mask[indices]
            m = mask.expand_as(img)
            img[m] = img_0[indices][m]
            label = label[indices]
            map[mask] = map_0[indices][mask]
        return img, label, map
