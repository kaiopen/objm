from typing import Any, Dict

from torch import nn

from .backbone import BACKBONE
from .neck import NECK
from .header import HEADER


class Model(nn.Module):
    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Dict[str, Any],
        header: Dict[str, Any],
        *args, **kwargs
    ) -> None:
        r'''

        #### Args:
        - backbone: the parameters to a backbone.
        - neck: the parameters to a neck.
        - header: the parameters to a header

        #### Methods;
        - forward

        '''
        super().__init__()
        self.backbone = BACKBONE[backbone.pop('type')](**backbone)
        self.neck = NECK[neck.pop('type')](**neck)
        self.header = HEADER[header.pop('type')](**header)

    def forward(self, x):
        return self.header(self.neck(self.backbone(x)))
