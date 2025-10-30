import json
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

from PIL import Image

try:
    from torchvision import transforms
except Exception:
    transforms = None


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
