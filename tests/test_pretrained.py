import torch

from astrolens.models.astropt import AstroPT
from astrolens.pretrained.astropt import load_pretrained_backbone

IMG_SIZE = 32
PATCH_SIZE = 8
N_LAYER, N_HEAD, N_EMBD = 2, 2, 16


def _fake_checkpoint():
    """A synthetic checkpoint matching the reference model's key layout,
    small enough to build in-memory instead of downloading a real one."""
    grid = IMG_SIZE // PATCH_SIZE
    num_patches = grid * grid
    sd = {
        "transformer.wpe.weight": torch.randn(num_patches, N_EMBD),
        "transformer.ln_f.weight": torch.randn(N_EMBD),
    }
    for i in range(N_LAYER):
        p = f"transformer.h.{i}."
        sd[p + "ln_1.weight"] = torch.randn(N_EMBD)
        sd[p + "attn.c_attn.weight"] = torch.randn(3 * N_EMBD, N_EMBD)
        sd[p + "attn.c_proj.weight"] = torch.randn(N_EMBD, N_EMBD)
        sd[p + "ln_2.weight"] = torch.randn(N_EMBD)
        sd[p + "mlp.c_fc.weight"] = torch.randn(4 * N_EMBD, N_EMBD)
        sd[p + "mlp.c_proj.weight"] = torch.randn(N_EMBD, 4 * N_EMBD)
    return {
        "model": {f"_orig_mod.{k}": v for k, v in sd.items()},
        "model_args": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD},
    }


def test_load_pretrained_backbone_maps_checkpoint_keys(tmp_path, monkeypatch):
    checkpoint = _fake_checkpoint()
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(path))

    model = AstroPT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=N_EMBD, depth=N_LAYER, heads=N_HEAD)
    model_args = load_pretrained_backbone(model, repo_id="fake/repo", filename="fake.pt")

    assert model_args == checkpoint["model_args"]
    torch.testing.assert_close(model.pos_embed.weight, checkpoint["model"]["_orig_mod.transformer.wpe.weight"])
    torch.testing.assert_close(model.ln_f.weight, checkpoint["model"]["_orig_mod.transformer.ln_f.weight"])
    torch.testing.assert_close(
        model.blocks[0].attn.qkv.weight, checkpoint["model"]["_orig_mod.transformer.h.0.attn.c_attn.weight"]
    )

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    model.eval()
    with torch.no_grad():
        model.forward_features(x)  # loaded weights run without shape errors
