from typing import Tuple, Union
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode

from kaitorch.data import tuple_2


def process_train(root: Path, dst: Path) -> None:
    src_box = root.joinpath('train', 'boxes')

    splits = []
    for line in tqdm(
        root.joinpath('train.txt').read_text().splitlines(), desc='TRAIN'
    ):
        name, label = line.split()

        f_box = src_box.joinpath(name).with_suffix('.xml')
        if f_box.exists():
            box = ET.parse(f_box).getroot().find('object').find('bndbox')
            box = (
                int(box.find('xmin').text), int(box.find('ymin').text),
                int(box.find('xmax').text) + 1, int(box.find('ymax').text) + 1
            )

        else:
            box = None

        splits.append((name, int(label), box))

    dst.joinpath('train').symlink_to(
        root.joinpath('train', 'images').resolve()
    )

    torch.save(splits, dst.joinpath('train.pt'))
    print('Number of images:', len(splits))


def process_val(
    root: Path,
    dst: Path,
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float, float] = (0.229, 0.224, 0.225)
):
    trans = transforms.Compose(
        (
            transforms.Resize(size, InterpolationMode.BICUBIC),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std)
        )
    )

    src = root.joinpath('val', 'images')
    h, w = size
    dst_img = dst.joinpath(f'val_{h}x{w}')

    splits = []
    for line in tqdm(
        root.joinpath('val.txt').read_text().splitlines(), desc='VAL'
    ):
        name, label = line.split()
        f = dst_img.joinpath(name).with_suffix('.pt')
        f.parent.mkdir(parents=True, exist_ok=Tuple)
        torch.save(trans(Image.open(src.joinpath(name)).convert('RGB')), f)

        splits.append((f.name, int(label)))

    torch.save(splits, dst.joinpath('val.pt'))
    print('Number of images:', len(splits))


def preprocess(
    root: Union[Path, str],
    split: str = 'train',
    size: Union[Tuple[int, int], int] = 224,
    mean: Tuple[float, float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float, float] = (0.229, 0.224, 0.225),
    dst: Union[Path, str] = Path.cwd().joinpath('tmp', 'ImageNet'),
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
