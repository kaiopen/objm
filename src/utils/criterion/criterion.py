from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from kaitorch.typing import TorchTensor, TorchFloat, TorchInt


def focal_loss(
    x: TorchTensor[TorchFloat],
    t: TorchTensor[TorchFloat]
) -> TorchTensor[TorchFloat]:
    r'''

    #### Args:
    - x: the prediction results from a model.
    - target: It should share the same shape with the prediction results.

    #### Returns:
    - a loss value.

    '''
    mask = -1 != t
    if torch.any(mask):
        return sigmoid_focal_loss(x[mask], t[mask], reduction='mean')
    return 0


class ClsSeg(nn.Module):
    def __init__(self, smoothing: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self._cls = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(
        self,
        x: Tuple[TorchTensor[TorchFloat], Sequence[TorchTensor[TorchFloat]]],
        t: Tuple[TorchTensor[TorchInt], TorchTensor[TorchInt]]
    ) -> Dict[str, Any]:
        x_cls, x_p = x
        t_cls, t_p = t

        loss_cls = self._cls(x_cls, t_cls.cuda())

        t_p = t_p.cuda()
        size = t_p.shape[2:]
        loss_seg = 0
        for x in x_p:
            loss_seg += focal_loss(
                F.interpolate(
                    x, size=size, mode='bilinear', align_corners=True
                ),
                t_p
            )

        loss = loss_cls + 0.5 * loss_seg
        return loss, {
            'loss': loss.item(),
            'cls': loss_cls.item(),
            'seg': loss_seg.item(),
        }
