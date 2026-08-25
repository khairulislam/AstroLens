"""Shared strong-lensing dark-matter-substructure classification dataset
utilities for AstroLens examples.

The canonical ML4SCI "Common Test I" benchmark (no substructure / CDM
point-mass subhalos / axion vortex, 150x150 single-channel), from
https://github.com/mwt5345/DeepLenseSim/tree/main/Model_I. The original
Google Drive links are dead (404); this loads a faithful re-upload,
`nelm/gsoc-2026-deeplense-dataset` on Hugging Face, verified to match the
expected class layout, image count, shape, and value range before use here.
"""

import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

REPO_ID = "nelm/gsoc-2026-deeplense-dataset"
CLASS_NAMES = ["no", "sphere", "vort"]  # no substructure / CDM / axion
NUM_CLASSES = len(CLASS_NAMES)

# computed over the full 3000-image train subset used by the training example
IMAGE_MEAN = 0.0617
IMAGE_STD = 0.1172


def _extract_subset(cache_dir: Path, split: str, max_per_class: int) -> None:
    """Extracts up to `max_per_class` .npy members per class for `split`,
    skipping any already on disk, without downloading the full 6.5 GB set."""
    zip_path = hf_hub_download(REPO_ID, "dataset.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as z:
        members = []
        for name in CLASS_NAMES:
            prefix = f"dataset/{split}/{name}/"
            existing = len(list((cache_dir / split / name).glob("*.npy"))) if (cache_dir / split / name).exists() else 0
            if existing >= max_per_class:
                continue
            class_members = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".npy"))
            members += class_members[:max_per_class]
        if members:
            z.extractall(cache_dir, members=members)


class LensingDataset(Dataset):
    """Map-style dataset over `<root>/dataset/<split>/<class_name>/*.npy`
    (150x150 single-channel float64 images, already scaled to [0, 1])."""

    def __init__(self, root, split: str, max_per_class: int, transform=None):
        root = Path(root)
        _extract_subset(root, split, max_per_class)
        self.transform = transform
        self.files = []
        for label, name in enumerate(CLASS_NAMES):
            paths = sorted((root / "dataset" / split / name).glob("*.npy"))[:max_per_class]
            self.files += [(p, label) for p in paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, label = self.files[i]
        image = torch.from_numpy(np.load(path)).float()
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def class_weights(dataset: LensingDataset) -> torch.Tensor:
    """Inverse-frequency class weights; the source dataset is class-balanced,
    so this is close to uniform but keeps parity with the other examples."""
    counts = torch.bincount(torch.tensor([label for _, label in dataset.files]), minlength=NUM_CLASSES).float()
    return counts.sum() / (NUM_CLASSES * counts)
