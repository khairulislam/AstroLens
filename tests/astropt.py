import torch

import astrolens
from astrolens.models.astropt import AstroPT

IMG_SIZE = 32
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2


def _tiny_model(num_classes=None):
    torch.manual_seed(0)
    model = AstroPT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        dim=16,
        depth=2,
        heads=2,
        num_classes=num_classes,
    )
    model.eval()
    return model


def test_registry_construction():
    assert astrolens.is_model("astropt")
    model = astrolens.create_model(
        "astropt", img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=16, depth=2, heads=2, num_classes=5
    )
    assert isinstance(model, AstroPT)


def test_forward_features_shape():
    model = _tiny_model()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    feats = model.forward_features(x)
    assert feats.shape == (2, NUM_PATCHES, 16)


def test_pretraining_forward_and_loss():
    model = _tiny_model()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    pred = model(x)
    assert pred.shape == (2, NUM_PATCHES, model.patch_dim)
    loss = model.loss(x)
    assert loss.dim() == 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_classification_head_output_shape():
    model = _tiny_model(num_classes=7)
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = model(x)
    assert y.shape == (2, 7)


def test_causal_embedding_does_not_see_future_patches():
    model = _tiny_model()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    x_perturbed = x.clone()
    x_perturbed[:, :, -PATCH_SIZE:, -PATCH_SIZE:] += 10.0  # perturb the last patch only

    with torch.no_grad():
        f0 = model.forward_features(x)
        f1 = model.forward_features(x_perturbed)

    torch.testing.assert_close(f0[:, :-1], f1[:, :-1])
    assert not torch.allclose(f0[:, -1], f1[:, -1])


def test_spiral_patchify_is_a_permutation_of_raster():
    model = _tiny_model()
    model_spiral = AstroPT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=16, depth=2, heads=2, spiral=True
    )
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    raster = model.patchify(x)
    spiral = model_spiral.patchify(x)
    assert spiral.shape == raster.shape
    raster_set = {tuple(p.tolist()) for p in raster[0]}
    spiral_set = {tuple(p.tolist()) for p in spiral[0]}
    assert raster_set == spiral_set
    assert not torch.equal(raster, spiral)


def test_draw_from_centre_differs_from_final_layer():
    model = _tiny_model()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        final = model.forward_features(x)
        centre = model.forward_features(x, draw_from_centre=True)
    assert centre.shape == final.shape
    assert not torch.allclose(final, centre)


def test_state_dict_save_load(tmp_path):
    model = _tiny_model(num_classes=5)
    path = tmp_path / "astropt.pt"
    torch.save(model.state_dict(), path)

    reloaded = _tiny_model(num_classes=5)
    reloaded.load_state_dict(torch.load(path))

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x))
