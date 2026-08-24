"""Shared utilities for loading a released Smith42/astroPT checkpoint into astrolens.models.astropt.AstroPT."""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

import astrolens

REPO_ID = "Smith42/astroPT"
CKPT_FILENAME = "models/fully_trained/0089M_params/030000_ckpt.pt"


def load_pretrained_backbone(model, repo_id: str, filename: str) -> dict:
    """Load the transformer body of a released AstroPT checkpoint into `model`.

    `Smith42/astroPT` checkpoints predate this library's multimodal-free
    reimplementation: they store the causal transformer under
    `transformer.h.*` (compiled with `torch.compile`, hence the
    `_orig_mod.` prefix to strip) with a patch encoder/decoder
    (`transformer.wte.*` / `lm_head.*`) too deep to line up with
    `AstroPT`'s single-layer patch projection. This transfers everything
    that does line up exactly — attention, MLP, layer norms, and the
    learned position embeddings (sliced to how many patches this image size
    needs) — and leaves the encoder, decoder, LoRA adapters, and
    classification head freshly initialized.

    Returns the checkpoint's `model_args` dict (n_layer, n_head, n_embd,
    patch_size, block_size, ...) so the caller can verify `model` was
    constructed with matching dimensions.
    """
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model_args = checkpoint["model_args"]

    num_patches = model.grid[0] * model.grid[1]
    mapped = {
        "pos_embed.weight": state_dict["transformer.wpe.weight"][:num_patches],
        "ln_f.weight": state_dict["transformer.ln_f.weight"],
    }
    for i in range(model_args["n_layer"]):
        src, dst = f"transformer.h.{i}.", f"blocks.{i}."
        mapped[dst + "ln_1.weight"] = state_dict[src + "ln_1.weight"]
        mapped[dst + "attn.qkv.weight"] = state_dict[src + "attn.c_attn.weight"]
        mapped[dst + "attn.proj.weight"] = state_dict[src + "attn.c_proj.weight"]
        mapped[dst + "ln_2.weight"] = state_dict[src + "ln_2.weight"]
        mapped[dst + "mlp.0.weight"] = state_dict[src + "mlp.c_fc.weight"]
        mapped[dst + "mlp.2.weight"] = state_dict[src + "mlp.c_proj.weight"]

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected, f"unexpected keys, checkpoint format may have changed: {unexpected}"
    return model_args


def load_pretrained_astropt(
    img_size: int,
    device,
    repo_id: str = REPO_ID,
    filename: str = CKPT_FILENAME,
    **model_kwargs,
):
    """Build an `astropt` model matching the checkpoint's dimensions and load its backbone.

    `model_kwargs` (e.g. `num_classes`, `lora_r`) are passed through to
    `astrolens.create_model`; `spiral=True` is always set, since released
    checkpoints were pretrained with spiral patch ordering.
    """
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    model_args = torch.load(path, map_location="cpu", weights_only=False)["model_args"]

    model = astrolens.create_model(
        "astropt",
        img_size=img_size,
        in_chans=model_args["n_chan"],
        patch_size=model_args["patch_size"],
        dim=model_args["n_embd"],
        depth=model_args["n_layer"],
        heads=model_args["n_head"],
        spiral=True,
        **model_kwargs,
    ).to(device)
    load_pretrained_backbone(model, repo_id, filename)
    return model


@torch.no_grad()
def compute_embeddings(model, loader, device) -> np.ndarray:
    """One mean-pooled embedding per image, from `AstroPT.forward_features`.

    Follows the reference model's `generate_embeddings`
    (`zss_..._mean_...npy` in its downstream scripts). `loader` may yield
    plain image batches or `(image, label)` pairs; only the images are used.
    """
    model.eval()
    embeddings = []
    for batch in tqdm(loader, desc="embedding", unit="batch"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        embeddings.append(model.forward_features(images.to(device)).mean(dim=1).cpu())
    return torch.cat(embeddings).numpy()
