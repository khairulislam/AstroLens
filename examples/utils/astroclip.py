"""Example-only helpers built on astrolens.pretrained.astroclip.

Fetches real g,r,z Legacy Survey cutouts by coordinate from the public
`legacysurvey.org` cutout service, already sized to AstroCLIP's expected
144x144 input, then applies `decals_to_rgb` — the arcsinh color-scaling
transform given verbatim in the AstroCLIP README
(https://github.com/PolymathicAI/AstroCLIP#processing-legacy-survey-images-directly)
for running the released model on cutouts outside its own premade dataset.
`UniverseTBD/mmu_gz10`'s `rgb_image` column (used elsewhere in these
examples) is not reused as model input for the same reason as
`utils/aion.py` — it isn't guaranteed to match this exact scaling.
"""

import numpy as np
import torch
from astropy.io import fits
from tqdm.auto import tqdm

CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
BANDS = ["g", "r", "z"]
CROP = 144

RGB_SCALES = {
    "u": (2, 1.5),
    "g": (2, 6.0),
    "r": (1, 3.4),
    "i": (0, 1.0),
    "z": (0, 2.2),
}


def decals_to_rgb(image: torch.Tensor, bands: list = BANDS, m: float = 0.03, Q: float = 20.0) -> torch.Tensor:
    """AstroCLIP's arcsinh color-scaling transform, verbatim from its README. `(B, C, H, W) -> (B, C, H, W)`."""
    axes, scales = zip(*[RGB_SCALES[b] for b in bands])
    scales = [scales[i] for i in axes]
    image = image.movedim(1, -1).flip(-1)
    scales = torch.tensor(scales, dtype=torch.float32, device=image.device)
    I = torch.sum(torch.clamp(image * scales + m, min=0), dim=-1) / len(bands)
    fI = torch.arcsinh(Q * I) / np.sqrt(Q)
    I = I + (I == 0.0) * 1e-6
    image = (image * scales + m) * (fI / I).unsqueeze(-1)
    image = torch.clamp(image, 0, 1)
    return image.movedim(-1, 1)


def fetch_legacysurvey_cutout(ra: float, dec: float, pixscale: float = 0.262, layer: str = "ls-dr10") -> np.ndarray:
    """Download one 3-band (g, r, z) flux cutout (shape `(3, CROP, CROP)`) centered at `(ra, dec)`."""
    url = f"{CUTOUT_URL}?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&bands={''.join(BANDS)}&size={CROP}"
    return fits.getdata(url).astype("float32")


def fetch_legacysurvey_cutouts(coords: list, **kwargs) -> tuple:
    """Download cutouts for a list of `(ra, dec)` pairs.

    Returns `(flux, kept)`: `flux` stacked to `(len(kept), 3, CROP, CROP)` and
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
def compute_image_embeddings(model, images: torch.Tensor, device, batch_size: int = 16) -> np.ndarray:
    """One AstroCLIP image embedding per cutout (already `decals_to_rgb`-preprocessed)."""
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="embedding", unit="batch"):
        batch = images[i : i + batch_size].to(device)
        embeddings.append(model(batch, input_type="image").cpu())
    return torch.cat(embeddings).numpy()
