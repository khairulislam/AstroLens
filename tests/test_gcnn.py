import torch

import astrolens
from astrolens.models.gcnn import GCNN, Group, transform_image, transform_kernel

IMG_SIZE = 255
TINY_FIELDS = [2] * 11


def _tiny_model(N=4, reflections=True, num_classes=5):
    torch.manual_seed(0)
    model = GCNN(
        N=N,
        reflections=reflections,
        img_size=IMG_SIZE,
        feature_fields=TINY_FIELDS,
        num_classes=num_classes,
    )
    model.eval()
    return model


def test_registry_construction():
    assert astrolens.is_model("gcnn_d4")
    model = astrolens.create_model(
        "gcnn_d4", img_size=IMG_SIZE, feature_fields=TINY_FIELDS, num_classes=5
    )
    assert isinstance(model, GCNN)


def test_forward_output_shape():
    model = _tiny_model(num_classes=7)
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (2, 7)


def test_backward_pass():
    model = _tiny_model()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_state_dict_save_load(tmp_path):
    model = _tiny_model()
    path = tmp_path / "gcnn.pt"
    torch.save(model.state_dict(), path)

    reloaded = _tiny_model()
    reloaded.load_state_dict(torch.load(path))

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x))


def test_feature_equivariance_before_group_pooling():
    model = _tiny_model()
    group = model.group
    channels = TINY_FIELDS[-1]

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        f0 = model.forward_features(x)
        for element in group.elements:
            xt = transform_image(x, group, element)
            ft = model.forward_features(xt)

            f0_spatial = transform_kernel(f0, group, element)
            f0_spatial = f0_spatial.reshape(1, channels, group.order, *f0.shape[-2:])
            perm = [group.index(group.mult(element, h)) for h in group.elements]
            expected = f0_spatial[:, :, perm].reshape(f0.shape)

            torch.testing.assert_close(ft, expected, atol=1e-4, rtol=1e-3)


def test_classification_invariance_after_group_pooling():
    model = _tiny_model()
    group = model.group

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        y0 = model(x)
        for element in group.elements:
            xt = transform_image(x, group, element)
            yt = model(xt)
            torch.testing.assert_close(y0, yt, atol=1e-4, rtol=1e-3)
