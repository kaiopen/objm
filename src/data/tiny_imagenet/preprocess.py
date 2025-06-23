from typing import Tuple, Union
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode

from kaitorch.data import tuple_2


def process_train(root: Path, dst: Path):
    src_img = root.joinpath('train')

    splits = []
    for line in root.joinpath('labels.txt').read_text().splitlines():
        c, label = line.split()
        label = int(label)

        for line in tqdm(
            src_img.joinpath(c, c + '_boxes.txt').read_text().splitlines(),
            desc=f'TRAIN {label}'
        ):
            name, x_0, y_0, x_1, y_1 = line.split()
            splits.append(
                (
                    c + '/images/' + name,
                    label,
                    (int(x_0), int(y_0), int(x_1) + 1, int(y_1) + 1)
                )
            )

    dst.joinpath('train').symlink_to(src_img.resolve())

    torch.save(splits, dst.joinpath('train.pt'))
    print('Number of images:', len(splits))


def process_val(
    root: Path,
    dst: Path,
    size: Tuple[int, int] = (64, 64),
    mean: Tuple[float, float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float, float] = (0.229, 0.224, 0.225)
):
    d = {}
    for line in root.joinpath('labels.txt').read_text().splitlines():
        c, label = line.split()
        d[c] = int(label)

    trans = transforms.Compose(
        (
            transforms.Resize(size, InterpolationMode.BICUBIC),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std)
        )
    )

    src_img = root.joinpath('val', 'images')
    h, w = size
    dst_img = dst.joinpath(f'val_{h}x{w}')

    splits = []
    for line in tqdm(
        root.joinpath('val', 'val_annotations.txt').read_text().splitlines(),
        desc='VAL'
    ):
        name, c = line.split()[:2]
        f = dst_img.joinpath(name).with_suffix('.pt')
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trans(Image.open(src_img.joinpath(name)).convert('RGB')), f)
        splits.append((f.name, d[c]))

    torch.save(splits, dst.joinpath('val.pt'))
    print('Number of images:', len(splits))


def preprocess(
    root: Union[Path, str],
    split: str = 'train',
    size: Union[Tuple[int, int], int] = 64,
    mean: Tuple[float, float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float, float] = (0.229, 0.224, 0.225),
    dst: Union[Path, str] = Path.cwd().joinpath('tmp', 'Tiny-ImageNet'),
    *args, **kwargs
):
    if isinstance(root, str):
        root = Path(root)
    if isinstance(dst, str):
        dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    match split:
        case 'train':
            process_train(root, dst)
        case 'val':
            process_val(root, dst, tuple_2(size), mean, std)
        case _:
            raise ValueError(
                'An invalid `split`. "train" or "val" is acceptable.'
            )
