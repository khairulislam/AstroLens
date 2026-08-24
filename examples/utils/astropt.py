"""Example-only helpers built on astrolens.pretrained.astropt."""

import numpy as np
import torch
from tqdm.auto import tqdm

from astrolens.pretrained.astropt import (  # noqa: F401  re-exported for existing notebook imports
    CKPT_FILENAME,
    REPO_ID,
    load_pretrained_astropt,
    load_pretrained_backbone,
)


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
