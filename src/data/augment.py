from typing import Callable, Optional, Sequence, Tuple, Union
import math
import random

from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
import torchvision.transforms.v2.functional as tvF

from kaitorch.typing import TorchTensor, TorchReal, real
from kaitorch.data import tuple_2
from kaitorch.zoo.cutmix import CutMix as _CutMix, cutmix_


def get_magnitude():
    return min(10, max(0, random.gauss(9, 0.5)))


def random_sign(a: real) -> real:
    if random.random() > 0.5:
        return -a
    return a


class AdjustBrightness:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.adjust_brightness(
            img, 1 + random_sign(get_magnitude() * 0.09)
        ) if random.random() < self._p else img, seg


class AdjustColor:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.adjust_saturation(
            img, 1 + random_sign(get_magnitude() * 0.09)
        ) if random.random() < self._p else img, seg


class AdjustContrast:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.adjust_contrast(
            img, 1 + random_sign(get_magnitude() * 0.09)
        ) if random.random() < self._p else img, seg


class AdjustSharpness:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.adjust_sharpness(
            img, 1 + random_sign(get_magnitude() * 0.09)
        ) if random.random() < self._p else img, seg


class AutoContrast(transforms.RandomAutocontrast):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def forward(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return super().forward(img), seg


class Equalize(transforms.RandomEqualize):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def forward(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return super().forward(img), seg


class Invert(transforms.RandomInvert):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def forward(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return super().forward(img), seg


class Posterize:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.posterize(
            img, int(8 - get_magnitude() * 0.4)
        ) if random.random() < self._p else img, seg


class Solarize:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        return tvF.solarize(
            img, 256 - get_magnitude() * 25.6
        ) if random.random() < self._p else img, seg


class Rotate:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self._p:
            v = random_sign(get_magnitude() * 0.3)
            return tvF.rotate(
                img,
                angle=v,
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            ), None if seg is None else tvF.rotate(
                seg,
                angle=v,
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            )
        return img, seg


class ShearX:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self._p:
            v = math.degrees(math.atan(random_sign(get_magnitude() * 0.03)))
            return tvF.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=[v, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                center=[0, 0]
            ), None if seg is None else tvF.affine(
                seg,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=[v, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                center=[0, 0]
            )
        return img, seg


class ShearY:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self._p:
            v = math.degrees(math.atan(random_sign(get_magnitude() * 0.03)))
            return tvF.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=[0, v],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                center=[0, 0]
            ), None if seg is None else tvF.affine(
                seg,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=[0, v],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                center=[0, 0]
            )
        return img, seg


class TranslateX:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self._p:
            v = int(random_sign(get_magnitude() * 0.045))
            return tvF.affine(
                img,
                angle=0,
                translate=[v * tvF.get_size(img)[1], 0],
                scale=1,
                shear=[0, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            ), None if seg is None else tvF.affine(
                seg,
                angle=0,
                translate=[v * tvF.get_size(seg)[1], 0],
                scale=1,
                shear=[0, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            )
        return img, seg


class TranslateY:
    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self._p:
            v = int(random_sign(get_magnitude() * 0.045))
            return tvF.affine(
                img,
                angle=0,
                translate=[0, v * tvF.get_size(img)[0]],
                scale=1,
                shear=[0, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            ), None if seg is None else tvF.affine(
                seg,
                angle=0,
                translate=[0, v * tvF.get_size(seg)[0]],
                scale=1,
                shear=[0, 0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0
            )
        return img, seg


class RandAugment:
    def __init__(self, p: float = 0.5) -> None:
        self._augs = (
            AdjustBrightness(p), AdjustColor(p), AdjustContrast(p),
            AdjustSharpness(p), AutoContrast(p), Equalize(p), Invert(p),
            Posterize(p), Solarize(p),
            Rotate(p), ShearX(p), ShearY(p), TranslateX(p), TranslateY(p)
        )
        self._weights = torch.ones(len(self._augs))

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        for i in torch.multinomial(self._weights, 2, True).tolist():
            img, seg = self._augs[i](img, seg)
        return img, seg


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p: float = 0.5, *args, **kwargs):
        super().__init__(p)

    def forward(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        if random.random() < self.p:
            return tvF.hflip(img), None if seg is None else tvF.hflip(seg)
        return img, seg


class RandomCropAndResize(transforms.RandomResizedCrop):
    r'''

    Here, we assume that the image and the segmentation map sharing the same
    size.

    '''
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        size: Optional[Union[Tuple[int, int], int]] = None
    ):
        super().__init__(
            size=(1, 1) if size is None else tuple_2(size),
            scale=scale,
            ratio=ratio
        )
        self._p = p

        if size is None:
            self.size = None

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        # Whether to crop the input.
        if random.random() < self._p:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            size = self.size or tvF.get_size(img)
            return tvF.resized_crop(
                img, i, j, h, w, size, InterpolationMode.BICUBIC
            ), None if seg is None else tvF.resized_crop(
                seg, i, j, h, w, size, InterpolationMode.BICUBIC
            )

        # Just resize.
        size = self.size
        if size is None:
            return img, seg

        return tvF.resize(
            img, size, InterpolationMode.BICUBIC
        ), None if seg is None else tvF.resize(
            seg, size, InterpolationMode.BICUBIC
        )


class Compose(transforms.Compose):
    def __init__(
        self,
        transforms: Sequence[
            Callable[
                [Image.Image, Image.Image], Tuple[Image.Image, Image.Image]
            ]
        ]
    ) -> None:
        super().__init__(transforms)

    def __call__(
        self, img: Image.Image, seg: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Union[Image.Image, None]]:
        for t in self.transforms:
            img, seg = t(img, seg)
        return img, seg


class CutMix(_CutMix):
    def __call__(
        self,
        img: TorchTensor[TorchReal],
        label: TorchTensor[TorchReal],
        seg: TorchTensor[TorchReal]
    ) -> Tuple[
        TorchTensor[TorchReal], TorchTensor[TorchReal], TorchTensor[TorchReal]
    ]:
        box, lam = self.random_box(*img.shape[2:])
        return cutmix_(img, box), self.mix_label(label, lam), cutmix_(seg, box)
