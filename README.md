<div align="center">
<h1>Objectness Scan for Efficient Vision Mamba</h1>

Kai Zhang([GitHub](https://github.com/kaiopen), [ORCID](https://orcid.org/0009-0005-9126-5139)), Xia Yuan([ORCID](https://orcid.org/0000-0002-7271-0058)), Chunxia Zhao

School of Computer Science and Engineering, Nanjing University of Science and Technology

Paper:

</div>

## Updates
- **`Jun. 23, 2025`**: Initial release. More codes are coming.

## Abstract

Mamba has made its debut in several visual tasks, but a more effective scanning strategy is still in need to unfold visual data into logical one-dimensional token sequences, ensuring spatial continuity and fully leveraging the advantages of Mamba in visual tasks. In this paper, an objectness scanning strategy with a dual-attention mechanism is proposed for efficient vision Mamba. Visual data is scanned according to its objectness. And the produced one-dimensional token sequences maintain visual spatial continuity while becoming compatible to the causal property of Mamba, owing to three built-in linguistic characteristics---dependency distance minimization, primacy effect and high accessibility. By incorporating the objectness scanning strategy and the dual-attention mechanism, the proposed vision Mamba model, ObjM, demonstrates significantly superiority in accurately identifying foreground objects and capturing both structural and detailed features. On both camouflaged object detection and salient object detection, it achieves comparable performance while reducing 20\% computational costs and 57\% parameters, by merely replacing the backbone of a state-of-the-art camouflaged object detection model. Furthermore, as a backbone, ObjM achieves a 71\% reduction in mean absolute error compared to ResNet-50, demonstrating excellent potential for segmentation.

## Overview

- ObjM serves as a vision Mamba backbone.
- Objectness Scanning Strategy

## Main Results

### Classification on ImageNet-1k

| Method | Input Size | GFLOPs | Params.(M) | Acc.(%) | Downloads |
|  :---: |    :---:   |  :---: |    :---:   |  :---:  |   :----:  |
|   ObjM | 224&times;224 | 2.9 | 10.1 | 78.1 | [objm.pt](./checkpoints/IN_objm.yaml/objm.pt) |

### Camouflaged Object Detection

<table>
    <tr>
        <th rowspan=2><center>Method</center></th>
        <th rowspan=2><center>Input Size</center></th>
        <th rowspan=2><center>GFLOPs</center></th>
        <th rowspan=2><center>Params.(M)</center></th>
        <th colspan=4><center>CAMO</center></th>
        <th colspan=4><center>COD10K</center></th>
        <th colspan=4><center>CN4K</center></th>
        <th rowspan=2><center>Downloads</center></th>
    </tr>
    <tr>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
    </tr>
    <tr align="center">
        <td>ObjM</td>
        <td>416&times;416</td><td>29.63</td><td>12.72</td>
        <td>0.061</td><td>0.783</td><td>0.841</td><td>0.895</td>
        <td>0.029</td><td>0.743</td><td>0.842</td><td>0.904</td>
        <td>0.041</td><td>0.803</td><td>0.863</td><td>0.912</td>
        <td><a href="./checkpoints/COD4040_objm.yaml/cod.pt">cod.pt</a></td>
    </tr>
</table>

### Salient Object Detection

<table>
    <tr>
        <th rowspan=2><center>Method</center></th>
        <th rowspan=2><center>Input Size</center></th>
        <th rowspan=2><center>GFLOPs</center></th>
        <th rowspan=2><center>Params.(M)</center></th>
        <th colspan=4><center>ECSSD</center></th>
        <th colspan=4><center>HLU-IS</center></th>
        <th colspan=4><center>DUTS-TE</center></th>
        <th rowspan=2><center>Downloads</center></th>
    </tr>
    <tr>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
        <th><center>MAE</center></th><th><center>F<sub>w</sub></center></th>
        <th><center>S</center></th><th><center>E</center></th>
    </tr>
    <tr align="center">
        <td>ObjM</td>
        <td>416&times;416</td><td>29.63</td><td>12.72</td>
        <td>0.027</td><td>0.929</td><td>0.933</td><td>0.931</td>
        <td>0.023</td><td>0.919</td><td>0.927</td><td>0.961</td>
        <td>0.027</td><td>0.874</td><td>0.906</td><td>0.919</td>
        <td><a href="./checkpoints/DUTS-TR_objm.yaml/sod.pt">sod.pt</a></td>
    </tr>
</table>

## Getting started
### Main Environments
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)

### Classification
#### Datasets
Download ImageNet-1k or Tiny-ImageNet datasets. Organize them with directory structures:
```
- data
    - ImageNet
        - train
            - boxes
                - n01440764
                    - n01440764_18.xml
                    - ...
                - ...
            - images
                - n01440764
                    - n01440764_18.JPEG
                    - ...
                - ...
        - val
            - boxes
                - ILSVRC2012_val_00000001.JPEG
                - ...
            - images
                - ILSVRC2012_val_00000001.JPEG
                - ...
        - train.txt
        - val.txt
    - Tiny-ImageNet
        - train
            - n01443537
                - images
                    - n01443537_0.JPEG
                    - ...
                - n01443537_boxes.txt
            - ...
        - val
            - images
                - val_0.JPEG
                - ...
            - val_annotations.txt
        - labels.txt
```

#### Preprocess

Command:
```shell
python tools/preprocess.py --config=<configuration file> --split=<split>
```

To preprocess the training set of ImageNet-1k:
```shell
python tools/preprocess.py --config="IN.yaml"
```

To preprocess the validation set of ImageNet-1k:
```shell
python tools/preprocess.py --config="IN.yaml" --split="val"
```

To preprocess the training set of Tiny-ImageNet:
```shell
python tools/preprocess.py --config="TIN.yaml"
```

To preprocess the validation set of Tiny-ImageNet:
```shell
python tools/preprocess.py --config="TIN.yaml" --split="val"
```

#### Training

Command:
```shell
torchrun --nproc_per_node=4 tools/train.py --config=<configuration file> --batch_size=<batch size> --num_worker=<number of dataloader workers> --end_epoch=<total epochs> --amp --cutmix --fgmix --lr=<initial learning rate>
```

To train on ImageNet-1k:
```shell
torchrun --nproc_per_node=4 tools/train.py --config="IN_objm.yaml" --batch_size=64 --num_worker=24 --end_epoch=300 --amp --cutmix --fgmix --lr=0.0001
```

To train on Tiny-ImageNet:
```shell
torchrun --nproc_per_node=4 tools/train.py --config="TIN_objm.yaml" --batch_size=64 --num_worker=24 --end_epoch=300 --amp --cutmix --fgmix --lr=0.0001
```

#### Evaluation

Command:
```shell
python tools/eval.py --config=<configuration file> --batch_size=<batch size> --num_worker=<number of dataloader workers> --start=<starting from which weight> --end=<end at which weight>
```

To evaluate on ImageNet-1k:
```shell
python tools/eval.py --config="IN_objm.yaml" --batch_size=1024 --num_worker=24 --start=200
```

To evaluate on Tiny-ImageNet:
```shell
python tools/eval.py --config="IN_objm.yaml" --batch_size=1024 --num_worker=24 --start=200
```
