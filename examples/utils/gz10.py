"""Shared UniverseTBD/mmu_gz10 (Galaxy10 DECals) dataset utilities for AstroLens examples."""

import io

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = [
    "disturbed",
    "merging",
    "round_smooth",
    "in_between_round_smooth",
    "cigar_shaped_smooth",
    "barred_spiral",
    "unbarred_tight_spiral",
    "unbarred_loose_spiral",
    "edge_on_no_bulge",
    "edge_on_with_bulge",
]
NUM_CLASSES = len(CLASS_NAMES)

# per-channel mean/std computed from a 2000-image sample of the gz10 train split
IMAGE_MEAN = [0.1675, 0.1625, 0.1586]
IMAGE_STD = [0.1288, 0.1178, 0.1109]


class GZ10Dataset(Dataset):
    """Map-style wrapper around an index subset of the HF gz10 split, applying transform lazily."""

    def __init__(self, hf_split, indices, transform, with_label=True):
        self.hf_split = hf_split
        self.indices = indices
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        example = self.hf_split[self.indices[i]]
        image = Image.open(io.BytesIO(example["rgb_image"]["bytes"])).convert("RGB")
        image = self.transform(image)
        return (image, example["gz10_label"]) if self.with_label else image


def load_split_702010(random_state=0):
    """Load gz10 and stratified-split indices 70% train / 10% val / 20% test."""
    from sklearn.model_selection import train_test_split

    gz10 = load_dataset("UniverseTBD/mmu_gz10", split="train")
    labels = gz10["gz10_label"]
    train_idx, rest_idx = train_test_split(
        range(len(gz10)), train_size=0.7, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=1 / 3,  # 1/3 of the remaining 30% -> 10% val, 20% test
        stratify=[labels[i] for i in rest_idx],
        random_state=random_state,
    )
    return gz10, labels, train_idx, val_idx, test_idx


def load_split_9010(random_state=0):
    """Load gz10 and a plain (unstratified) 90/10 train/val split, for label-free pretraining."""
    gz10 = load_dataset("UniverseTBD/mmu_gz10", split="train")
    generator = torch.Generator().manual_seed(random_state)
    perm = torch.randperm(len(gz10), generator=generator).tolist()
    split_at = int(0.9 * len(gz10))
    return gz10, perm[:split_at], perm[split_at:]


def class_weights(labels, indices, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights from a label subset, to counter GZ10's class imbalance."""
    counts = torch.bincount(
        torch.tensor([labels[i] for i in indices]), minlength=num_classes
    ).float()
    return counts.sum() / (num_classes * counts)


def run_classification_epoch(model, loader, criterion, device, train, optimizer=None):
    """One train/eval pass over `loader` for a classifier; returns (avg_loss, accuracy)."""
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += images.size(0)

    return total_loss / count, correct / count
