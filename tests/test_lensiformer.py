import torch

import astrolens
from astrolens.models.lensiformer import Lensiformer

IMG_SIZE = 16
PATCH_SIZE = 8


def _tiny_model(num_classes=5):
    torch.manual_seed(0)
    model = Lensiformer(
        img_size=IMG_SIZE,
        in_chans=1,
        patch_size=PATCH_SIZE,
        embed_dim=8,
        num_heads=2,
        encoder_depth=1,
        decoder_depth=1,
        num_classes=num_classes,
    )
    model.eval()
    return model


def test_registry_construction():
    assert astrolens.is_model("lensiformer")
    model = astrolens.create_model(
        "lensiformer", img_size=IMG_SIZE, in_chans=1, patch_size=PATCH_SIZE, embed_dim=8, num_classes=5
    )
    assert isinstance(model, Lensiformer)


def test_forward_output_shape():
    model = _tiny_model(num_classes=7)
    x = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (2, 7)


def test_backward_pass():
    model = _tiny_model()
    x = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
    y = model(x)
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_state_dict_save_load(tmp_path):
    model = _tiny_model()
    path = tmp_path / "lensiformer.pt"
    torch.save(model.state_dict(), path)

    reloaded = _tiny_model()
    reloaded.load_state_dict(torch.load(path))

    x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x))
