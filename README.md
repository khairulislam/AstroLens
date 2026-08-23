# AstroLens

A unified library of vision models for astronomy.

## Table of Contents

- [Usage](#usage)
- [Astroformer](#astroformer)
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

![astroformer model overview](images/astroformer.png)

This [ICLR 2023 workshop paper](https://arxiv.org/abs/2304.05350) proposes a hybrid transformer-convolutional architecture, to learn efficiently on less amount of data, inspired by CoAtNet and MaxViT. First down-sampling the feature map via a multi-stage network with gradual pooling to reduce the spatial size and then employing the global relative attention. Found that the C-C-C-T design works much better than C-C-T-T which was adapted as the layout for CoAtNet. This is due to high generalization capability and training stability. 

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

## Resources

## Citations

```
@article{dagli2023astroformer,
  title={Astroformer: More Data Might Not be All You Need for Classification},
  author={Dagli, Rishit},
  journal={arXiv preprint arXiv:2304.05350},
  year={2023}
}

```