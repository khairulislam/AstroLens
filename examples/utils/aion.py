"""Example-only helpers built on astrolens.pretrained.aion.

Fetches real calibrated 4-band (g, r, i, z) Legacy Survey cutouts by
coordinate from the public `legacysurvey.org` cutout service, following the
same approach as AION's own tutorial
(https://github.com/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb):
AION's image codec expects flux in physical units, not RGB pixel values, so
`UniverseTBD/mmu_gz10`'s `rgb_image` column (used elsewhere in these
examples) is not reused here — only its `ra`/`dec` columns, as coordinates
to look up real flux cutouts.
"""

import numpy as np
import torch
from astropy.io import fits
from tqdm.auto import tqdm

CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]


def fetch_legacysurvey_cutout(ra: float, dec: float, pixscale: float = 0.262, layer: str = "ls-dr10") -> np.ndarray:
    """Download one 4-band flux cutout (shape `(4, H, W)`) centered at `(ra, dec)`."""
    url = f"{CUTOUT_URL}?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}"
    return fits.getdata(url).astype("float32")


def fetch_legacysurvey_cutouts(coords: list, **kwargs) -> tuple:
    """Download cutouts for a list of `(ra, dec)` pairs.

    Returns `(flux, kept)`: `flux` stacked to `(len(kept), 4, H, W)` and
    `kept` the indices into `coords` that succeeded, so a caller can index
    any per-coordinate metadata (labels, thumbnails) to stay aligned. Skips
    and warns on any coordinate the cutout service fails to serve (e.g.
    outside the Legacy Survey DR10 footprint) rather than aborting the whole
    batch.
    """
    cutouts, kept = [], []
    for idx, (ra, dec) in enumerate(tqdm(coords, desc="fetching cutouts", unit="ex")):
        try:
            cutouts.append(fetch_legacysurvey_cutout(ra, dec, **kwargs))
            kept.append(idx)
        except Exception as e:
            print(f"skipping ({ra}, {dec}): {e}")
    return np.stack(cutouts), kept


@torch.no_grad()
def compute_image_embeddings(model, codec_manager, flux: np.ndarray, device, batch_size: int = 16, num_encoder_tokens: int = 600) -> np.ndarray:
    """One mean-pooled AION embedding per cutout, from the image-only modality."""
    from aion.modalities import LegacySurveyImage

    embeddings = []
    for i in tqdm(range(0, len(flux), batch_size), desc="embedding", unit="batch"):
        batch = torch.tensor(flux[i : i + batch_size], device=device)
        image = LegacySurveyImage(flux=batch, bands=BANDS)
        tokens = codec_manager.encode(image)
        embeddings.append(model.encode(tokens, num_encoder_tokens=num_encoder_tokens).mean(dim=1).cpu())
    return torch.cat(embeddings).numpy()
