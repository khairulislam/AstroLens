"""Shared Smith42/galaxies dataset utilities for AstroLens examples.

`Smith42/galaxies` (https://huggingface.co/datasets/Smith42/galaxies) is the
dataset the AstroPT reference authors' own pretraining and downstream
(`linear_probe.py`, `scripts/euclid/downstream_tasks/`) scripts use: DESI
Legacy Survey galaxy cutouts, unlabeled but carrying continuous photometric
properties (magnitudes, redshift, stellar mass) and debiased Galaxy Zoo
vote-fraction columns, in place of GZ10's single discrete class label. It's
large enough that examples here stream only the first `n` and materialize
them into memory, rather than downloading the full dataset.
"""

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import Dataset

REPO_ID = "Smith42/galaxies"
REVISION = "v2.0"


class GalaxiesDataset(Dataset):
    """Map-style wrapper around a list of pre-fetched Smith42/galaxies examples."""

    def __init__(self, examples: list, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.transform(self.examples[i]["image"].convert("RGB"))


def load_galaxies(n: int, columns: list, filter_key: str = None, split: str = "train") -> list:
    """Stream and materialize the first `n` examples of `columns` from Smith42/galaxies.

    If `filter_key` is given, only examples where that column is not None
    are kept (matching the reference `linear_probe.py`'s approach of
    filtering to a property before sampling).
    """
    dataset = load_dataset(REPO_ID, split=split, revision=REVISION, streaming=True).select_columns(columns)
    if filter_key is not None:
        dataset = dataset.filter(lambda example: example[filter_key] is not None)
    return list(tqdm(dataset.take(n), total=n, desc="streaming galaxies", unit="ex"))


def split_9010(examples: list, random_state: int = 0) -> tuple:
    """Plain (unstratified) 90/10 index split, for label-free pretraining."""
    generator = torch.Generator().manual_seed(random_state)
    perm = torch.randperm(len(examples), generator=generator).tolist()
    split_at = int(0.9 * len(examples))
    return perm[:split_at], perm[split_at:]
