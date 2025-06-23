from typing import Sequence, Tuple, Union
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tvF

from kaitorch.typing import TorchTensor, TorchFloat, TorchInt8, TorchInt64, \
    TorchReal
from kaitorch.data import tuple_2

from ..dataset import Dataset as _Dataset
from ..augment import Compose, CutMix, RandAugment, \
    RandomHorizontalFlip, RandomCropAndResize
from ..foreground_mix import ForegroundMix


class Dataset(_Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str = 'train',
        num_category: int = 1000,
        size: Union[Tuple[int, int], int] = 224,
        mean: Tuple[float, float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float, float] = (0.229, 0.224, 0.225),
        cutmix: bool = False,
        fgmix: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(root, str):
            root = Path(root)

        size = tuple_2(size)
        h, w = size

        match split:
            case 'train':
                self._root = root.joinpath('train')
                self._data = torch.load(
                    root.joinpath('train.pt'), weights_only=False
                )

                self._aug = Compose(
                    (
                        RandomHorizontalFlip(),
                        RandAugment(),
                        RandomCropAndResize(size=size)
                    )
                )

                self._aug_img = transforms.Compose(
                    (
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean, std),
                        transforms.RandomErasing(0.25, value='random')
                    )
                )
                self._aug_seg = transforms.Compose(
                    (
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True)
                    )
                )

                self._fgmix = ForegroundMix(t=0, p=0.5) if fgmix else None

                self._cutmix = CutMix(num_category=num_category) \
                    if cutmix else None

                self._get = self._get_train
                self._collate = self._collate_train

            case 'val':
                h, w = tuple_2(size)
                self._root = root.joinpath(f'val_{h}x{w}')
                self._data = torch.load(
                    root.joinpath('val.pt'), weights_only=False
                )

                self._get = self._get_val
                self._collate = self._collate_val

            case _:
                raise ValueError(
                    'An invalid `split`. "train" or "val" is acceptable.'
                )

        self._len = len(self._data)
        self.__i = 0

    def _get_train(
        self, index: int
    ) -> Tuple[TorchTensor[TorchFloat], Tuple[int, TorchTensor[TorchInt8]]]:
        img, label, box = self._data[index]
        img = Image.open(self._root.joinpath(img)).convert('RGB')

        # Generate a map.
        if box is None:
            img, _ = self._aug(img)
            seg = torch.full((1, *tvF.get_size(img)), -1, dtype=torch.float)
        else:
            seg = torch.zeros((1, *tvF.get_size(img)), dtype=torch.float)
            x_0, y_0, x_1, y_1 = box
            seg[:, y_0: y_1, x_0: x_1] = 1  # (1, H, W)
            seg = tvF.to_pil_image(seg)

            img, seg = self._aug(img, seg)
            seg = self._aug_seg(seg)

        return self._aug_img(img), (label, seg)

    def _collate_train(
        self,
        x: Sequence[
            Tuple[TorchTensor[TorchReal], Tuple[int, TorchTensor[TorchReal]]]
        ]
    ) -> Tuple[
        TorchTensor[TorchReal],
        Tuple[TorchTensor[TorchInt64], TorchTensor[TorchReal]]
    ]:
        imgs = []
        labels = []
        segs = []
        for img, (label, seg) in x:
            imgs.append(img)
            labels.append(label)
            segs.append(seg)

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)
        segs = torch.stack(segs, dim=0)

        if (mix := self._fgmix) is not None:
            imgs, labels, segs = mix(imgs, labels, segs)

        if (mix := self._cutmix) is not None:
            imgs, labels, segs = mix(imgs, labels, segs)

        return imgs, (labels, segs)
        # (B, 3, H, W) (B,) (B, 1, H, W)

    def _get_val(
        self, index: int
    ) -> Tuple[TorchTensor[TorchFloat], int]:
        img, label = self._data[index]
        return torch.load(self._root.joinpath(img), weights_only=False), label

    def _collate_val(
        self, x: Sequence[Tuple[TorchTensor[TorchReal], int]]
    ) -> Tuple[TorchTensor[TorchReal], TorchTensor[TorchInt64]]:
        imgs = []
        labels = []
        for img, label in x:
            imgs.append(img)
            labels.append(label)
        return torch.stack(imgs, dim=0), torch.tensor(labels)

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[TorchTensor[TorchFloat], Tuple[int, TorchTensor[TorchInt8]]],
        Tuple[TorchTensor[TorchFloat], int]
    ]:
        return self._get(index)

    def __len__(self):
        return self._len

    def __next__(self):
        if (i := self.__i) < self._len:
            data = self[i]
            self.__i += 1
            return data
        self.__i = 0
        raise StopIteration

    def collate(
        self,
        x: Union[
            Sequence[
                Tuple[
                    TorchTensor[TorchReal],
                    Tuple[int, TorchTensor[TorchReal]]
                ]
            ],
            Sequence[Tuple[TorchTensor[TorchReal], int]]
        ]
    ) -> Union[
        Tuple[
            TorchTensor[TorchReal],
            Tuple[TorchTensor[TorchInt64], TorchTensor[TorchReal]]
        ],
        Tuple[TorchTensor[TorchReal], TorchTensor[TorchInt64]]
    ]:
        return self._collate(x)
