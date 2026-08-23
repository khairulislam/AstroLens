import torch

import astrolens
from astrolens.models.astroformer import AstroFormer

IMG_SIZE = 64


def _tiny_model(num_classes=5):
    torch.manual_seed(0)
    model = AstroFormer(img_size=IMG_SIZE, num_classes=num_classes)
    model.eval()
    return model


def test_registry_construction():
    assert astrolens.is_model("astroformer")
    model = astrolens.create_model("astroformer", img_size=IMG_SIZE, num_classes=5)
    assert isinstance(model, AstroFormer)


def test_forward_output_shape():
    model = _tiny_model(num_classes=7)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (1, 7)


def test_num_classes_zero_returns_pooled_features():
    model = _tiny_model(num_classes=0)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (1, model.head_hidden_size)


def test_backward_pass():
    model = _tiny_model()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
