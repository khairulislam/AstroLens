# AstroLens

A unified library of vision models for astronomy.

## Table of Contents

- [Usage](#usage)
- [Astroformer](#astroformer)
- [Linformer](#linformer)
- [Resources](#resources)
- [Citations](#citations)

## Usage

```bash
pip install -r requirements.txt
```

```python
import astrolens

astrolens.list_models()
model = astrolens.create_model("astroformer", num_classes=10)
```


## Astroformer

<img src="images/astroformer.png" alt="astroformer model framework" width="800">

This [ICLR 2023 workshop paper](https://arxiv.org/abs/2304.05350) proposes a hybrid transformer-convolutional architecture, to learn efficiently on less amount of data, inspired by CoAtNet and MaxViT. First down-sampling the feature map via a multi-stage network with gradual pooling to reduce the spatial size and then employing the global relative attention. Found that the C-C-C-T design works much better than C-C-T-T which was adapted as the layout for CoAtNet. This is due to high generalization capability and training stability. Achieved 94.86% on [Galaxy10 DECals](https://huggingface.co/datasets/mwalmsley/galaxy10_decals).

```python
import torch
from astrolens.models.astroformer import AstroFormer

model = AstroFormer(img_size=256, in_chans=3, num_classes=10)
img = torch.randn(2, 3, 256, 256)
logits = model(img)
```

Or through the registry:

```python
import astrolens

model = astrolens.create_model("astroformer", num_classes=10)
```

## Linformer

<img src="images/linformer.png" alt="Linformer model framework" width="600">


This [paper](https://arxiv.org/abs/2110.01024) applies a Vision Transformer to galaxy morphology classification, replacing standard quadratic self-attention with [Linformer](https://arxiv.org/abs/2006.04768)'s linear-complexity low-rank attention approximation for efficiency. Achieved 80.55% overall accuracy on an 8-class [Galaxy Zoo 2](https://data.galaxyzoo.org/) dataset.

```python
import torch
from astrolens.models.linformer import Linformer

model = Linformer(img_size=224, in_chans=3, num_classes=8)
img = torch.randn(2, 3, 224, 224)
logits = model(img)
```

Or through the registry:

```python
import astrolens

model = astrolens.create_model("linformer", num_classes=8)
```

## Resources

## Citations

```
@article{dagli2023astroformer,
  title={Astroformer: More Data Might Not be All You Need for Classification},
  author={Dagli, Rishit},
  journal={arXiv preprint arXiv:2304.05350},
  year={2023}
}

@article{lin2021galaxy,
  title={Galaxy Morphological Classification with Efficient Vision Transformer},
  author={Lin, Joshua Yao-Yu and Liao, Song-Mao and Huang, Hung-Jin and Kuo, Wei-Ting and Ou, Olivia Hsuan-Min},
  journal={arXiv preprint arXiv:2110.01024},
  year={2021}
}

```