from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from torch import nn

from kaitorch.typing import TorchTensor, TorchFloat, TorchReal
from kaitorch.nn.conv import Conv2dBlock
from kaitorch.zoo.hrnet import Transition, Fusion
from kaitorch.zoo.resnet import Bottleneck

from ..utils.proposer import MultiScaleProposer
from ..utils.former import Former2d
from ..utils.mfam import MFAM


class Stem(nn.Module):
    r'''

    #### Args:
    - in_channels
    - out_channels: the number of output channels.
    - padding_mode: `zeros`, `reflect`, `replicate`, `circular` or their
        combination working on dimension `H` and `W` of the input. Its length
        should be less than or equal to 2 if it is a sequence.
    - activation: `relu`, `leakyrelu` or other activation.
    - activation_kw: arguments of activation.

    ### Methods:
    - forward

    ## forward
    #### Args:
    - x: feature map. Its shape should be `(B, in_channels, H, W)`.

    #### Returns:
    - Feature map. Its shape is `(B, out_channels, H // 4, W // 4)`.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: Union[str, Sequence[str]] = 'zeros',
        activation: str = 'relu',
        activation_kw: Dict[str, Any] = {'inplace': True},
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._stem = nn.Sequential(
            Conv2dBlock(
                in_channels, out_channels, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            ),
            Conv2dBlock(
                out_channels, out_channels, 3, 2, 1,
                padding_mode=padding_mode,
                activation=activation,
                activation_kw=activation_kw
            )
        )

    def forward(self, x: TorchTensor[TorchFloat]) -> TorchTensor[TorchFloat]:
        return self._stem(x)


class SingleStage(nn.Module):
    r'''

    #### Args:
    - blocks: parameters for each block. There should be a key "type"
        indicating the type of the block.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: a feature map. Its shape should be `(B, C, H, W)`.

    #### Returns:
    - List of a feature map. Its length is `1`. The shape of the feature map is
        `(B, C', H, W)`.

    '''
    def __init__(
        self, blocks: Sequence[Dict[str, Any]], *args, **kwargs
    ) -> None:
        super().__init__()
        self._branch = nn.Sequential(
            *[Bottleneck(**block) for block in blocks]
        )

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> List[TorchTensor[TorchFloat]]:
        return [self._branch(x)]


class Branches(nn.Module):
    r'''

    #### Args:
    - in_channels
    - former
    - depth
    - scans: scan directions.
    - drops: probabilities for drop path.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    #### Returns:
    - List of feature maps. Its length is `N`. The shapes of the feature maps
        are `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    '''
    def __init__(
        self,
        in_channels: Sequence[int],
        former: Dict[str, Any],
        depth: int,
        scans: Union[str, Sequence[str]] = 'H+',
        drops: Union[float, Sequence[float]] = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__()
        if isinstance(scans, str):
            scans = [scans] * depth
        elif (d := len(scans)) != depth:
            raise ValueError(f'The length of `scans` ({d}) should be {depth}.')

        if isinstance(drops, float):
            drops = [drops] * depth
        elif (d := len(drops)) != depth:
            raise ValueError(f'The length of `drops` ({d}) should be {depth}.')

        self._branches = nn.ModuleList()
        for c in in_channels:
            self._branches.append(
                nn.Sequential(
                    *[
                        Former2d(in_channels=c, scan=s, drop=p, **former)
                        for s, p in zip(scans, drops)
                    ]
                )
            )

    def forward(
        self, xs: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        for i, (x, f) in enumerate(zip(xs, self._branches)):
            xs[i] = f(x)
        return xs


class GlobalStage(nn.Module):
    r'''A stage of the HRNet.

    #### Args:
    - transition
    - branches
    - fusion
    - drops: probabilities for drop path.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    ### Returns:
    - List of feature maps. If `M >= N`, the length of the list is `M`. The
        shapes of the feature maps are `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{M - 1}, H // 2^M, W // 2^M)`
        where `C_0`, `C_1`, ..., `C_(M-1)` are in `out_channels`. If `M < N`,
        the length of the list is `N`. The shapes of the feature maps are
        `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{M - 1}, H // 2^M, W // 2^M)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    '''
    def __init__(
        self,
        transition: Dict[str, Any],
        branches: Sequence[Sequence[Sequence[Dict[str, Any]]]],
        fusion: Dict[str, Any],
        drops: Union[float, Sequence[float]] = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        depth = 0
        depths = []
        for kw in branches:
            d = kw['depth']
            depths.append(d)
            depth += d

        if isinstance(drops, float):
            drops = [drops] * depth
        elif (d := len(drops)) != depth:
            raise ValueError(f'The length of `drops` ({d}) should be {depth}.')

        ms = [Transition(**transition)]
        i = 0
        for kw, d in zip(
            branches, torch.cumsum(torch.as_tensor(depths), dim=0).tolist()
        ):
            ms += [Branches(drops=drops[i: d], **kw), Fusion(**fusion)]
            i = d
        self._ms = nn.Sequential(*ms)

    def forward(
        self, xs: List[TorchTensor[TorchFloat]]
    ) -> List[TorchTensor[TorchFloat]]:
        return self._ms(xs)


class ObjBranches(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        former: Dict[str, Any],
        depth: int,
        k: float = 0.8,
        drops: Union[float, Sequence[float]] = 0.
    ):
        super().__init__()
        if isinstance(drops, float):
            drops = [drops] * depth
        elif (d := len(drops)) != depth:
            raise ValueError(f'The length of `drops` ({d}) should be {depth}.')

        self._ms = nn.ModuleList(
            [
                MFAM(
                    in_channels=c,
                    former=former,
                    depth=depth,
                    k=k,
                    drops=drops
                ) for c in in_channels
            ]
        )

    def forward(
        self,
        xs: List[TorchTensor[TorchFloat]],
        proposals: Sequence[TorchTensor[TorchReal]]
    ) -> List[TorchTensor[TorchFloat]]:
        for i, (x, p, m) in enumerate(zip(xs, proposals, self._ms)):
            xs[i] = m(x + x * torch.sigmoid(p), p)
        return xs


class ObjStage(nn.Module):
    r'''A stage of the HRNet.

    #### Args:
    - transition
    - proposer
    - branches
    - fusion
    - drops: probabilities for drop path.

    #### Methods:
    - forward

    ## forward
    #### Args:
    - x: sequence of feature maps. Its length should be `N`. The shapes of the
        feature maps should be `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{N - 1}, H // 2^N, W // 2^N)`
        where `C_0`, `C_1`, ..., `C_(N-1)` are in `in_channels`.

    ### Returns:
    - List of feature maps. If `M >= N`, the length of the list is `M`. The
        shapes of the feature maps are `(B, C_0, H, W)`,
        `(B, C_1, H // 2, W // 2)`, ..., `(B, C_{M - 1}, H // 2^M, W // 2^M)`
        where `C_0`, `C_1`, ..., `C_(M-1)` are in `out_channels`. If `M < N`,
        the length of the list is `N`. The shapes of the feature maps are
        `(B, C_0, H, W)`, `(B, C_1, H // 2, W // 2)`, ...,
        `(B, C_{M - 1}, H // 2^M, W // 2^M)`, ...,
        `(B, C_{N - 1}, H // 2^N, W // 2^N)`.

    '''
    def __init__(
        self,
        transition: Dict[str, Any],
        proposer: Dict[str, Any],
        branches: Dict[str, Any],
        fusion: Dict[str, Any],
        drops: Union[float, Sequence[float]] = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._trans = Transition(**transition)
        self._proposer = MultiScaleProposer(**proposer)
        self._branches = ObjBranches(drops=drops, **branches)
        self._fus = Fusion(**fusion)

    def forward(
        self, xs: List[TorchTensor[TorchFloat]]
    ) -> Tuple[List[TorchTensor[TorchFloat]], List[TorchTensor[TorchFloat]]]:
        xs = self._trans(xs)
        ps = self._proposer(xs)
        return self._fus(self._branches(xs, ps)), ps


class ObjM(nn.Module):
    def __init__(
        self,
        stem: Dict[str, Any],
        stages: Sequence[Dict[str, Any]],
        drop: float = 0.,
        *args, **kwargs
    ) -> None:
        super().__init__()
        self._stem = Stem(**stem)

        depths = []
        for stage in stages[1: -1]:
            depths.append(sum([kw['depth'] for kw in stage['branches']]))
        depths.append(stages[-1]['branches']['depth'])
        drops = [p.item() for p in torch.linspace(0, drop, sum(depths))]
        depths = torch.cumsum(torch.as_tensor(depths), dim=0).tolist()

        self._stages = nn.Sequential()
        self._stages.append(SingleStage(**stages[0]))

        i = 0
        for stage, d in zip(stages[1: -1], depths):
            self._stages.append(GlobalStage(drops=drops[i: d], **stage))
            i = d

        self._obj = ObjStage(drops=drops[i: depths[-1]], **stages[-1])

        self.apply(self.init_weights)

    def forward(
        self, x: TorchTensor[TorchFloat]
    ) -> Tuple[List[TorchTensor[TorchFloat]], List[TorchTensor[TorchFloat]]]:
        return self._obj(self._stages(self._stem(x)))

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
