import json
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from PIL import Image

try:
    from torchvision import transforms
except Exception:
    transforms = None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
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


class PatchSelectionDataset(Dataset):
    """
    Dataset that maps CIFAR100/HF images and JSONL records of selected patch indices
    to inputs and multi-hot targets.

    Each JSONL line must include:
      - image_id: int (index into the HF dataset split)
      - selected_indices: List[int] (indices in [0, num_patches-1])
    """

    def __init__(
        self,
        jsonl_path: str,
        dataset_name: str = "cifar100",
        split: str = "test",
        num_classes: Optional[int] = None,
        img_size: int = 32,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.records = _load_jsonl(jsonl_path)
        if not self.records:
            raise ValueError(f"No records found in {jsonl_path}")

        # Load HF dataset split (expects 'img' field)
        self.ds = load_dataset(dataset_name, split=split)

        # Determine output dimension (num patches)
        if num_classes is None:
            max_idx = 0
            for r in self.records:
                sel = r.get("selected_indices", [])
                if sel:
                    max_idx = max(max_idx, max(sel))
            self.num_classes = int(max_idx) + 1
        else:
            self.num_classes = int(num_classes)

        # Build transforms
        if transform is not None:
            self.transform = transform
        else:
            if transforms is None:
                raise ImportError("torchvision not available; please provide a custom transform")
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_id = rec["image_id"]
        selected = rec.get("selected_indices", [])

        sample = self.ds[image_id]
        img = sample["image"]  # PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        x = self.transform(img)
        if isinstance(x, torch.Tensor):
            x = x.clone().detach().contiguous()  # ensure resizable storage for DataLoader

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for s in selected:
            if 0 <= s < self.num_classes:
                y[s] = 1.0

        return x, y


