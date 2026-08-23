import torch

import astrolens
from astrolens.models.linformer import Linformer

IMG_SIZE = 28
PATCH_SIZE = 14


def _tiny_model(num_classes=5):
    torch.manual_seed(0)
    model = Linformer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=16, depth=2, heads=2, k=4, num_classes=num_classes
    )
    model.eval()
    return model


def test_registry_construction():
    assert astrolens.is_model("linformer")
    model = astrolens.create_model(
        "linformer", img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=16, depth=2, heads=2, k=4, num_classes=5
    )
    assert isinstance(model, Linformer)


def test_forward_output_shape():
    model = _tiny_model(num_classes=7)
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (2, 7)


def test_num_classes_zero_returns_pooled_features():
    model = _tiny_model(num_classes=0)
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (2, 16)


def test_backward_pass():
    model = _tiny_model()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_state_dict_save_load(tmp_path):
    model = _tiny_model()
    path = tmp_path / "linformer.pt"
    torch.save(model.state_dict(), path)

    reloaded = _tiny_model()
    reloaded.load_state_dict(torch.load(path))

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x))
