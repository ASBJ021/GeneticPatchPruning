
from datasets import load_dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
try:
    from torchvision import transforms
except Exception:
    transforms = None
from typing import Tuple
import os
import sys
from pathlib import Path

import json
from typing import List, Dict, Any, Optional, Tuple

import torch

from PIL import Image
import random
from collections import defaultdict



PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f'{PROJECT_ROOT = }')
from gpp.dataset.dataset_patch_selection import PatchSelectionDataset




def load_data(DATASET_NAME, NUM_SAMPLES):
    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    names = ds.features["fine_label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts


def load_data_folder(DATA_DIR, NUM_SAMPLES, SPLIT="validation", cache_dir= "/home/vault/v123be/v123be15/hf_cache"):
    # cache_dir = '/home/vault/v123be/v123be15/hf_cache'
    train_files = {"train": DATA_DIR + "train-*.parquet"}
    test_files = {"test": DATA_DIR + "test-*.parquet"}
    val_files = {"validation": DATA_DIR + "validation-*.parquet"}
    split = SPLIT
    print(f'Loading {split} data from {DATA_DIR}')
    if split == "train":
        ds = load_dataset("parquet", data_files=train_files, split=split, cache_dir=cache_dir)
    elif split == "test":
        ds = load_dataset("parquet", data_files=test_files, split=split, cache_dir=cache_dir)
    elif split == "val" or split == "validation":
        ds = load_dataset("parquet", data_files=val_files, split=split, cache_dir=cache_dir)

    if NUM_SAMPLES > 0:
        ds = ds.select(range(NUM_SAMPLES))


    print(ds[0])

    names = ds.features["label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]



    return ds, prompts


def load_data_folder_mixed_samples(DATA_DIR, NUM_SAMPLES, SPLIT="test", seed=42):
    """
    Load a parquet ImageNet dataset split and optionally select samples from different classes.
    """
    random.seed(seed)
    train_files = {"train": f"{DATA_DIR}/train-*.parquet"}
    test_files  = {"test": f"{DATA_DIR}/test-*.parquet"}
    val_files   = {"validation": f"{DATA_DIR}/validation-*.parquet"}

    print(f"Loading {SPLIT} data from {DATA_DIR}")
    if SPLIT == "train":
        ds = load_dataset("parquet", data_files=train_files, split="train")
    elif SPLIT in ["test"]:
        ds = load_dataset("parquet", data_files=test_files, split="test")
    elif SPLIT in ["val", "validation"]:
        ds = load_dataset("parquet", data_files=val_files, split="validation")
    else:
        raise ValueError(f"Unknown split: {SPLIT}")

    # If NUM_SAMPLES > 0, select samples distributed across classes
    if NUM_SAMPLES > 0:
        labels = ds["label"]
        n_classes = len(ds.features["label"].names)
        per_class = max(1, NUM_SAMPLES // n_classes)

        class_to_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            class_to_indices[lbl].append(i)

        selected_indices = []
        for lbl, idxs in class_to_indices.items():
            if len(idxs) > per_class:
                selected_indices.extend(random.sample(idxs, per_class))
            else:
                selected_indices.extend(idxs)

        # If we got fewer than NUM_SAMPLES (because of rounding or few samples per class),
        # pad with random remaining indices
        if len(selected_indices) < NUM_SAMPLES:
            remaining = list(set(range(len(ds))) - set(selected_indices))
            extra = random.sample(remaining, min(len(remaining), NUM_SAMPLES - len(selected_indices)))
            selected_indices.extend(extra)

        ds = ds.select(selected_indices)

    print("Example:", ds[0])
    print(f"Loaded {len(ds)} samples from {len(ds.features['label'].names)} classes.")

    # Create CLIP-style text prompts
    names = ds.features["label"].names
    prompts = [f"a photo of a {n.replace('_', ' ')}" for n in names]

    return ds, prompts
    
def load_data_normal(DATASET_NAME, NUM_SAMPLES, SPLIT="test"):
    ds = load_dataset(DATASET_NAME,split = SPLIT)
    # ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))
    if NUM_SAMPLES > 0:
        ds = ds.select(range(NUM_SAMPLES))
    print(ds[0])
    names = ds.features["label"].names
    prompts = [f"a photo of a {n.replace('_',' ')}" for n in names]
    return ds, prompts




def plot_dataset(ds, prompts):
    """
    Plot images one by one from a HuggingFace dataset.
    
    Args:
        ds: HuggingFace dataset (must contain "img" and "fine_label")
        prompts: optional list of prompt strings (e.g., class names)
        max_images: limit how many images to plot (default None = all)
    """
    n = len(ds) 
    
    for i in range(n):
        sample = ds[i]
        print(sample)
        img = sample["image"]    # already a PIL Image (HF datasets return PIL for image columns)
        label = sample["label"]
        print(label)
        print(f'{img.size = }')

        plt.imshow(img)
        
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


def build_dataloaders(
    jsonl_path: str,
    dataset_name: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, int]:
    t = None
    if transforms is not None:
        t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    full_ds = PatchSelectionDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        num_classes=num_classes,
        img_size=img_size,
        transform=t,
    )

    n_total = len(full_ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f'Loading {dataset_name}')
    print(f'Number of training samples: {n_train}')
    print(f'Number of validation samples: {n_val}')
    print(f'Total number of samples: {n_total}')
    print(f'Number of classes: {full_ds.num_classes}')
    return train_loader, val_loader, full_ds.num_classes


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return records


class PatchIndexDataset(Dataset):
    """
    Wraps a HuggingFace dataset (images) together with JSONL annotations
    of selected patch indices to produce (image_tensor, multi_hot_targets).

    Each JSONL line must include:
      - image_id: int (index into the dataset)
      - selected_indices: List[int]

    Targets are constructed as a multi-hot vector of length `num_classes`.
    """

    def __init__(
        self,
        ds,
        jsonl_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        img_size: int = 32,
        transform: Optional[Any] = None,
        items: Optional[List[Tuple[int, List[int]]]] = None,
        return_index: bool = False,
    ) -> None:
        """
        If `items` is provided, it should be a list of tuples (ds_index, selected_indices).
        Otherwise, provide `jsonl_path` and the dataset will be filtered by matching
        dataset indices with annotation image_id via enumerate.
        """
        super().__init__()
        self.ds = ds
        self.return_index = return_index

        if items is None:
            if not jsonl_path:
                raise ValueError("Either items or jsonl_path must be provided")
            records = _load_jsonl(jsonl_path)
            if not records:
                raise ValueError(f"No records found in {jsonl_path}")

            # Build map from image_id -> selected_indices (last occurrence wins)
            id_to_sel: Dict[int, List[int]] = {}
            for r in records:
                iid = int(r.get("image_id"))
                sel = r.get("selected_indices", []) or []
                id_to_sel[iid] = sel

            # Enumerate dataset and select only items whose index is in annotations
            # if not isinstance(img, Image.Image):
            #     img = Image.fromarray(img)
            # img = img.convert("RGB")

            filtered: List[Tuple[int, List[int]]] = []
            for idx, _ in enumerate(ds):
                if idx in id_to_sel:
                    filtered.append((idx, id_to_sel[idx]))
            self.items = filtered
        else:
            self.items = items

        if not self.items:
            raise ValueError("No matching items between dataset and annotations")

        # Determine number of classes (patch count)
        if num_classes is None:
            max_idx = 0
            for _, sel in self.items:
                if sel:
                    max_idx = max(max_idx, max(sel))
            self.num_classes = int(max_idx) + 1
        else:
            self.num_classes = int(num_classes)

        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            if transforms is None:
                raise ImportError("torchvision not available; please provide a transform")
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds_index, selected = self.items[idx]

        sample = self.ds[ds_index]
        if isinstance(sample, dict):
            img = sample.get("image") or sample.get("img")
        else:
            img = None
        if img is None:
            raise KeyError("Sample does not contain 'image' or 'img' keys")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        x = self.transform(img)
        if isinstance(x, torch.Tensor):
            x = x.clone()

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for s in selected:
            if 0 <= s < self.num_classes:
                y[s] = 1.0

        if self.return_index:
            return x, y, ds_index
        return x, y


def split_dataset(
    full_ds: PatchIndexDataset,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """
    Split PatchIndexDataset into train/val/test according to provided ratios.
    Returns three new PatchIndexDataset instances sharing the same underlying HF dataset
    and transforms, but with disjoint items.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    n = len(full_ds.items)
    if n == 0:
        raise ValueError("Cannot split empty dataset")

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    def pick(indices):
        items = [full_ds.items[i] for i in indices]
        return PatchIndexDataset(
            ds=full_ds.ds,
            jsonl_path=None,
            num_classes=full_ds.num_classes,
            img_size=full_ds.transform.transforms[0].size[0] if hasattr(full_ds.transform, 'transforms') else 32,
            transform=full_ds.transform,
            items=items,
        )

    return pick(idx_train), pick(idx_val), pick(idx_test)

def main():
    dataset_name = "ILSVRC/imagenet-1k"
    data_dir = "/home/vault/v123be/v123be15/imagenet-1k/data/"
    train_files = {"train": data_dir + "train-*.parquet"}
    test_files = {"test": data_dir + "test-*.parquet"}
    val_files = {"validation": data_dir + "validation-*.parquet"}

    split = "validation"
    cache_dir = '/home/vault/v123be/v123be15/hf_cache'

    # num_samples = 10
    # if split == "train":
    #     ds = load_dataset("parquet", data_files=train_files, split=split)
    # elif split == "test":
    #     ds = load_dataset("parquet", data_files=test_files, split=split)
    # elif split == "val" or split == "validation":
    #     ds = load_dataset("parquet", data_files=val_files, split=split)

    # ds = load_dataset("parquet", data_files=test_files, split=split)
    
    # features = ds.features
    # classes = ds.features["label"].names
    # print(f'{classes.shape = }')
    # print(f'{classes = }')
    # if "label" in features and hasattr(features["label"], "names"):
    #     # HF mapping (often WNIDs or human names depending on how it was built)
    #     hf_idx_to_name = features["label"].names
    #     print(f'{hf_idx_to_name = }')

    ds = load_data_folder(data_dir, NUM_SAMPLES=100, SPLIT=split, cache_dir=cache_dir)
    # mixed_ds = load_data_folder_mixed_samples(data_dir, NUM_SAMPLES=100, SPLIT=split)
    print(ds[0])
    print(ds)

    # ds, _ = load_data_normal(dataset_name, num_samples, split)
    # print(ds.shape)



if __name__ == "__main__":
    main()